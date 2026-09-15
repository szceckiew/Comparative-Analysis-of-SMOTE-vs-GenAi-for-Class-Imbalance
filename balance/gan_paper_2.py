import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    """
    Generator network from Figure 5.
    Hidden Layers: 4 (128 -> 256 -> 512 -> 1024 -> output_dim)
    Activation: ReLU
    Normalization: Batch Normalization
    """

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
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network from Figure 5.
    Hidden Layers: 3 (512 -> 256 -> 128 -> 1)
    Activation: LeakyReLU
    Normalization: None
    """

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


class GBO:
    """
    GAN-Based Oversampling (GBO).

    Uses a standard GAN to generate synthetic minority class samples.
    Follows Algorithm 2 and parameter settings from Figure 5 of the paper.

    Parameters
    ----------
    noise_dim : int
        Dimension of the noise vector z fed to the generator.
    learning_rate : float
        Learning rate for both generator and discriminator (default: 0.00001).
    epochs : int
        Number of training epochs (T in Algorithm 2).
    batch_size : int
        Mini-batch size for training (default: 32 as per paper).
    patience : int
        Number of epochs with no improvement before early stopping.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        noise_dim=64,
        learning_rate=0.00001,
        epochs=500,
        batch_size=32,
        patience=50,
        device=None,
    ):
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = None
        self.discriminator = None

    def fit(self, X_minority):
        """
        Train the GAN on minority class samples.

        Parameters
        ----------
        X_minority : np.ndarray
            Minority class samples, shape (n_samples, n_features).
        """
        data_dim = X_minority.shape[1]

        self.generator = Generator(self.noise_dim, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim).to(self.device)

        optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )

        criterion = nn.BCEWithLogitsLoss()

        real_data = torch.FloatTensor(X_minority).to(self.device)
        dataset = torch.utils.data.TensorDataset(real_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        best_g_loss = float("inf")
        epochs_no_improve = 0
        best_g_state = None

        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            n_batches = 0

            for (batch_real,) in dataloader:
                current_batch_size = batch_real.size(0)

                label_real = torch.ones(current_batch_size, 1).to(self.device)
                label_fake = torch.zeros(current_batch_size, 1).to(self.device)

                optimizer_d.zero_grad()

                output_real = self.discriminator(batch_real)
                loss_real = criterion(output_real, label_real)

                z = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(z).detach()
                output_fake = self.discriminator(fake_data)
                loss_fake = criterion(output_fake, label_fake)

                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()

                z = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                fake_data = self.generator(z)
                output_fake = self.discriminator(fake_data)
                loss_g = criterion(output_fake, label_real)

                loss_g.backward()
                optimizer_g.step()

                d_loss_epoch += loss_d.item()
                g_loss_epoch += loss_g.item()
                n_batches += 1

            avg_g_loss = g_loss_epoch / n_batches

            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                epochs_no_improve = 0
                best_g_state = {
                    k: v.clone() for k, v in self.generator.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                self.generator.load_state_dict(best_g_state)
                break

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.epochs}] "
                    f"D_loss: {d_loss_epoch / n_batches:.4f} "
                    f"G_loss: {avg_g_loss:.4f}"
                )

    def sample(self, n_samples):
        """
        Generate synthetic minority samples using the trained generator.

        Parameters
        ----------
        n_samples : int
            Number of synthetic samples to generate (nfake in Algorithm 2).

        Returns
        -------
        np.ndarray
            Generated samples, shape (n_samples, n_features).
        """
        if self.generator is None:
            raise RuntimeError("GBO has not been fitted yet. Call fit() first.")

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.noise_dim).to(self.device)
            synthetic = self.generator(z).cpu().numpy()
        return synthetic

    def fit_resample(self, X, y):
        """
        Fit the GAN on minority class and resample to balance the dataset.

        Parameters
        ----------
        X : np.ndarray
            Full feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Labels, shape (n_samples,).

        Returns
        -------
        X_resampled : np.ndarray
            Balanced feature matrix.
        y_resampled : np.ndarray
            Balanced labels.
        """
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        majority_count = counts.max()

        X_resampled = X.copy()
        y_resampled = y.copy()

        for cls in classes:
            if cls == majority_class:
                continue

            X_minority = X[y == cls]
            n_to_generate = majority_count - len(X_minority)

            if n_to_generate <= 0:
                continue

            print(
                f"GBO: Generating {n_to_generate} samples for class {cls} "
                f"(minority: {len(X_minority)}, majority: {majority_count})"
            )

            self.fit(X_minority)
            X_synthetic = self.sample(n_to_generate)

            X_resampled = np.vstack([X_resampled, X_synthetic])
            y_resampled = np.concatenate(
                [y_resampled, np.full(n_to_generate, cls)]
            )

        return X_resampled, y_resampled