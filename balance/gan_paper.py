import numpy as np
import pandas as pd


# --- GAN-based oversampling function ---
def apply_gan_paper_oversample(x_train, y_train, epochs=200, noise_dim=100, batch_size=64, device=None):
    """
    GAN based on the paper: arXiv:2210.12870 – Credit Card Fraud Detection using GANs.
    Automatically detects the minority class and generates samples for it.
    """
    # Lazy imports
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.utils import shuffle

    # --- Network Definitions (Scoped inside the function) ---
    class Generator(nn.Module):
        def __init__(self, noise_dim, output_dim):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(noise_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),

                nn.Linear(1024, output_dim),
                nn.Sigmoid()
                # NOTE: Sigmoid output requires feature normalization in range [0, 1]
            )

        def forward(self, z):
            return self.model(z)

    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LeakyReLU(0.2),

                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),

                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),

                nn.Linear(128, 1)
                # No terminal Sigmoid activation since BCEWithLogitsLoss is utilized
            )

        def forward(self, x):
            return self.model(x)

    # --- Function Logic ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    # --- 1. Automatic Minority Class Detection ---
    classes, counts = np.unique(y_train, return_counts=True)

    # Locate class with minimum sample representation
    minority_idx = np.argmin(counts)
    majority_idx = np.argmax(counts)

    minority_label = classes[minority_idx]
    majority_label = classes[majority_idx]

    n_minority = counts[minority_idx]
    n_majority = counts[majority_idx]

    # Determine required number of synthetic samples
    n_to_generate = n_majority - n_minority

    print(
        f" GAN Logic: Majority Class: {majority_label} ({n_majority}), Minority Class: {minority_label} ({n_minority})")

    # --- 2. Balance Verification ---
    if n_to_generate <= 0:
        print(" Data is already balanced (or logic error). Returning original data.")
        return x_train, y_train

    print(f" GAN oversampling: generating {n_to_generate} samples for class {minority_label}")

    # Isolate minority class data partition for GAN optimization
    minority_data = x_train[y_train == minority_label]
    input_dim = x_train.shape[1]

    # Initialize networks
    g = Generator(noise_dim, input_dim).to(device)
    d = Discriminator(input_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(g.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(d.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # --- 3. Train GAN ---
    g.train()
    d.train()

    for epoch in range(epochs):
        # Sample mini-batch from ground-truth minority class instances
        # Modulo indexing safely handles configurations where batch_size > n_minority
        idx = np.random.randint(0, n_minority, batch_size)
        real_samples = torch.tensor(minority_data[idx]).to(device)

        # One-sided label smoothing
        real_labels = (torch.ones((batch_size, 1)) * 0.9).to(device)
        fake_labels = (torch.zeros((batch_size, 1)) + 0.1).to(device)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_samples = g(z)

        d_real = d(real_samples)
        d_fake = d(fake_samples.detach())  # Detach graph to avoid propagating gradients to G
        loss_d = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # --- Train Generator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_samples = g(z)
        d_fake = d(fake_samples)
        # Generator objective: deceive discriminator (target label 1.0)
        loss_g = criterion(d_fake, torch.ones((batch_size, 1)).to(device))

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

    # --- 4. Generate New Samples ---
    g.eval()  # Switch to evaluation mode (critical for BatchNorm layers)
    n_batches = int(np.ceil(n_to_generate / batch_size))
    generated = []

    for _ in range(n_batches):
        z = torch.randn(batch_size, noise_dim).to(device)
        with torch.no_grad():
            fake = g(z).cpu().numpy()
        generated.append(fake)

    if not generated:
        print(" Warning: No samples generated.")
        return x_train, y_train

    generated = np.vstack(generated)[:n_to_generate]

    # Combine datasets
    x_balanced = np.vstack([x_train, generated])
    # Assign target minority class label to generated instances
    y_generated = np.full(len(generated), minority_label)
    y_balanced = np.hstack([y_train, y_generated])

    # x_balanced, y_balanced = shuffle(x_balanced, y_balanced, random_state=42)

    return x_balanced, y_balanced