import numpy as np


def apply_wgangp_oversample(x_train, y_train, epochs=100, batch_size=64, noise_dim=30, n_critic=3, lambda_gp=15):
    """
    Trains WGAN-GP for tabular oversampling.
    Default hyperparameters set according to literature benchmarks:
    - n_critic (critic updates per generator step) = 3
    - lambda_gp (gradient penalty weight) = 15
    - noise_dim = 30
    """
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # --- 1. Neural Network Architectures ---
    class Generator(nn.Module):
        def __init__(self, noise_dim, input_dim):
            super(Generator, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, input_dim)
            )

        def forward(self, z):
            return self.net(z)


    class Critic(nn.Module):
        def __init__(self, input_dim):
            super(Critic, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LayerNorm(128),  
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)


    # --- 2. Gradient Penalty Calculation ---
    def compute_gradient_penalty(D, real_samples, fake_samples, device):
        alpha = torch.rand(real_samples.size(0), 1).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(real_samples.size(0), 1).to(device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # --- Core Pipeline Execution ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running WGAN-GP on device: {device}")

    unique, counts = np.unique(y_train, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_count = np.max(counts)
    minority_count = np.min(counts)
    n_to_generate = majority_count - minority_count

    if n_to_generate <= 0:
        print("Data is already balanced. Skipping GAN augmentation.")
        return x_train, y_train

    print(f"Minority class: {minority_class}. Generating {n_to_generate} samples.")

    minority_data = x_train[y_train == minority_class]
    dataset = TensorDataset(torch.FloatTensor(minority_data))

    if len(minority_data) < batch_size:
        curr_batch_size = len(minority_data)
        drop_last = False
    else:
        curr_batch_size = batch_size
        drop_last = True

    dataloader = DataLoader(dataset, batch_size=curr_batch_size, shuffle=True, drop_last=drop_last)
    input_dim = x_train.shape[1]

    generator = Generator(noise_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

    generator.train()
    critic.train()

    for epoch in range(epochs):
        for i, (real_imgs,) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            current_bs = real_imgs.size(0)

            optimizer_C.zero_grad()
            z = torch.randn(current_bs, noise_dim).to(device)
            fake_imgs = generator(z)

            real_validity = critic(real_imgs)
            fake_validity = critic(fake_imgs)

            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_C.step()

            if i % n_critic == 0:
                optimizer_G.zero_grad()
                fake_imgs = generator(z)
                fake_validity = critic(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()

    print("Training completed successfully.")

    generator.eval()
    generated_samples = []
    batches_to_gen = int(np.ceil(n_to_generate / batch_size))

    with torch.no_grad():
        for _ in range(batches_to_gen):
            z = torch.randn(batch_size, noise_dim).to(device)
            gen_batch = generator(z).cpu().numpy()
            generated_samples.append(gen_batch)

    generated_data = np.vstack(generated_samples)[:n_to_generate]
    generated_labels = np.full((n_to_generate,), minority_class)

    x_balanced = np.vstack([x_train, generated_data])
    y_balanced = np.hstack([y_train, generated_labels])

    return x_balanced, y_balanced