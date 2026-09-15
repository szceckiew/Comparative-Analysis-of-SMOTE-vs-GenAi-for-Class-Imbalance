import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# --- GAN-based oversampling function ---
def apply_gan_oversample(x_train, y_train, epochs=300, noise_dim=16, batch_size=128, device=None):
    """
    Generates synthetic minority class samples using a simple GAN.
    Automatically detects the minority class.
    """
    # Lazy imports (prevents freeze on launch)
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # --- Network Definitions (Scoped inside the function) ---
    class Generator(nn.Module):
        def __init__(self, noise_dim, output_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(noise_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
                # Note: If input features are normalized to [0, 1], append nn.Sigmoid() here
            )

        def forward(self, z):
            return self.model(z)

    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    # --- Core Pipeline Execution ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train = np.asarray(x_train, dtype=np.float32)
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
        f"GAN Logic: Majority Class: {majority_label} ({n_majority}), Minority Class: {minority_label} ({n_minority})")

    # --- 2. Balance Verification ---
    if n_to_generate <= 0:
        print("Data is already balanced. Returning original data.")
        return x_train, y_train

    print(f"GAN oversampling: generating {n_to_generate} samples for class {minority_label}")

    # Isolate minority class data partition for GAN training
    minority_data = x_train[y_train == minority_label]
    input_dim = x_train.shape[1]

    # Initialize networks
    G = Generator(noise_dim, input_dim).to(device)
    D = Discriminator(input_dim).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.001)
    optimizer_D = optim.Adam(D.parameters(), lr=0.001)

    # --- 3. Train GAN ---
    G.train()
    D.train()

    for epoch in range(epochs):
        # Sample mini-batch
        idx = np.random.randint(0, n_minority, batch_size)
        real_samples = torch.tensor(minority_data[idx]).to(device)

        # Target labels
        real_labels = torch.ones((batch_size, 1)).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_samples = G(z)

        D_real = D(real_samples)
        D_fake = D(fake_samples.detach())
        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_samples = G(z)
        D_fake = D(fake_samples)
        loss_G = criterion(D_fake, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    # --- 4. Generate New Samples ---
    G.eval()
    n_batches = int(np.ceil(n_to_generate / batch_size))
    generated = []

    for _ in range(n_batches):
        z = torch.randn(batch_size, noise_dim).to(device)
        with torch.no_grad():
            fake = G(z).cpu().numpy()
        generated.append(fake)

    if not generated:
        return x_train, y_train

    generated = np.vstack(generated)[:n_to_generate]

    # Combine datasets
    X_balanced = np.vstack([x_train, generated])
    # Assign target minority class label to generated instances
    y_generated = np.full(len(generated), minority_label)
    y_balanced = np.hstack([y_train, y_generated])

    X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

    print("New class distribution after GAN oversampling:")
    print(pd.Series(y_balanced).value_counts())

    return X_balanced, y_balanced