import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# --- Global Settings ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- 3. Training and Oversampling Function (Keras/TF implementation) ---
def train_and_augment_gan_keras(X_train, y_train, epochs=10, batch_size=32):
    """
    Trains the GAN in Keras/TensorFlow on minority class samples and generates
    new samples to balance the dataset.
    Automatically detects the minority class.
    """
    # --- Lazy imports ---
    import tensorflow as tf
    from tensorflow import keras
    
    # Deferred random seed initialization for TF until invocation
    tf.random.set_seed(RANDOM_SEED)

    # --- 1. Generator (Nested architecture to prevent global Keras namespace pollution) ---
    def build_generator(coding_size, n_features):
        """Creates the Keras Generator model."""
        generator = keras.models.Sequential([
            # Input layer expects a noise vector of size 'coding_size'
            keras.layers.Dense(100, activation='selu', input_shape=[coding_size]),
            keras.layers.Dense(200, activation='selu'),
            keras.layers.Dense(300, activation='selu'),
            keras.layers.Dense(400, activation='selu'),
            keras.layers.Dense(500, activation='selu'),
            # Output layer matches the number of features (n_features)
            # Sigmoid activation scales the output to [0, 1]
            keras.layers.Dense(n_features, activation='sigmoid')
        ], name="Generator")
        return generator

    # --- 2. Discriminator (Nested architecture) ---
    def build_discriminator(n_features):
        """Creates the Keras Discriminator model."""
        discriminator = keras.models.Sequential([
            # Input layer receives the real/generated data
            keras.layers.Dense(n_features, input_shape=[n_features]),
            keras.layers.Dense(500, activation='selu'),
            keras.layers.Dense(400, activation='selu'),
            keras.layers.Dense(300, activation='selu'),
            keras.layers.Dense(200, activation='selu'),
            keras.layers.Dense(100, activation='selu'),
            # Output layer for binary classification (Real/Fake)
            keras.layers.Dense(1, activation='sigmoid')
        ], name="Discriminator")
        return discriminator

    # --- Core Pipeline Execution ---
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    # --- 1. Automatic Minority Class Detection ---
    classes, counts = np.unique(y_train, return_counts=True)

    minority_idx = np.argmin(counts)
    majority_idx = np.argmax(counts)

    minority_label = classes[minority_idx]
    majority_label = classes[majority_idx]

    n_minority = counts[minority_idx]
    n_majority = counts[majority_idx]

    n_to_generate = n_majority - n_minority

    print(
        f" Keras GAN Logic: Majority Class: {majority_label} ({n_majority}), Minority Class: {minority_label} ({n_minority})")

    # --- 2. Balance Verification ---
    if n_to_generate <= 0:
        print(" Data is already balanced. Returning original data.")
        return X_train, y_train

    # Identify and isolate the minority class samples
    minority = X_train[y_train == minority_label]

    n_features = X_train.shape[1]
    coding_size = n_features // 2  # Logic from tutorial
    if coding_size < 1: coding_size = 1  # Safety check for low dim data

    print(
        f" GAN oversampling: generating {n_to_generate} samples (coding_size={coding_size}) for class {minority_label}")

    # --- Build and Compile Models ---
    generator = build_generator(coding_size, n_features)
    discriminator = build_discriminator(n_features)

    # 1. Compile Discriminator (trainable)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(learning_rate=10 ** -4))

    # 2. Build full GAN model (Discriminator is frozen for Generator training)
    discriminator.trainable = False
    gan = keras.models.Sequential([generator, discriminator], name="GAN")
    gan.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=10 ** -4))

    # --- GAN Training Loop ---
    print(f"Starting GAN training for {epochs} epochs...")

    idxs_minor_train = np.array(range(n_minority))
    n_batch = int(n_minority // batch_size)

    for epoch in range(epochs):
        # Shuffle minority indices for each epoch
        np.random.RandomState(seed=RANDOM_SEED).shuffle(idxs_minor_train)

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        # Handle small datasets where n_batch might be 0
        current_n_batch = max(n_batch, 1)

        for i in range(current_n_batch):
            if n_batch == 0:
                # Take all samples if dataset is smaller than batch_size
                mb = idxs_minor_train
            else:
                first_idx = i * batch_size
                last_idx = min((i + 1) * batch_size, n_minority)
                mb = idxs_minor_train[first_idx: last_idx]

            if len(mb) == 0: continue

            # --- Train Discriminator (D) ---

            # Get real samples for this batch
            real_features = minority[mb, :]

            # Generate fake samples
            noise = tf.random.normal(shape=[len(mb), coding_size])
            gen_features = generator(noise)

            # Combine and label samples: [0.]*fake + [1.]*real
            gen_real_features = tf.concat([gen_features, real_features], axis=0)
            y_disc = tf.constant([[0.]] * len(mb) + [[1.]] * len(mb))

            # Train D (unfrozen)
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(gen_real_features, y_disc)

            # --- Train Generator (G) ---

            # Generate new noise vector
            noise = tf.random.normal(shape=[len(mb), coding_size])
            # G wants its fake samples to be labeled as "real" (1)
            y_gen = tf.constant([[1.]] * len(mb))

            # Train G via the full GAN model (D is frozen)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss

        if (epoch + 1) % 2 == 0 or epoch == 0:
            avg_d_loss = epoch_d_loss / current_n_batch
            avg_g_loss = epoch_g_loss / current_n_batch
            print(f"Epoch {epoch + 1}/{epochs} | Avg. D_Loss: {avg_d_loss:.4f}, Avg. G_Loss: {avg_g_loss:.4f}")

    # --- Generate new samples ---
    print(f"Generating {n_to_generate} samples...")

    # Create the required noise vector
    noise_vector = tf.random.normal(shape=[n_to_generate, coding_size])

    generated_features = generator.predict(noise_vector, verbose=0)

    # Ensure generated data is within the normalized range [0, 1]
    generated_features = np.clip(generated_features, 0, 1)

    # --- Combine and Shuffle ---
    # Assign detected minority label to newly synthesized samples
    generated_labels = np.full(len(generated_features), minority_label)

    x_balanced = np.vstack([X_train, generated_features])
    y_balanced = np.hstack([y_train, generated_labels])

    return x_balanced, y_balanced


if __name__ == '__main__':
    # --- EXAMPLE USAGE (Simulated BCW Data) ---
    n_features = 30
    n_majority = 178
    n_minority = 106

    X_sample = np.random.rand(n_majority + n_minority, n_features).astype(np.float32)
    # Test with reversed labels (0 is minority) to verify detection logic
    y_sample = np.concatenate([np.ones(n_majority), np.zeros(n_minority)]).astype(int)

    X_sample, y_sample = shuffle(X_sample, y_sample, random_state=RANDOM_SEED)

    print(f"Original dataset size: {X_sample.shape}")
    print(f"Class distribution: {pd.Series(y_sample).value_counts().to_dict()}")

    # Run GAN training and augmentation
    X_balanced, y_balanced = train_and_augment_gan_keras(
        X_sample,
        y_sample,
        epochs=10,
        batch_size=32
    )

    print("\n--- RESULT ---")
    print(f"Balanced dataset size: {X_balanced.shape}")
    print(f"Balanced class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")