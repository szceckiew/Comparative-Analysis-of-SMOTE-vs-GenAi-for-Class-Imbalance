import numpy as np
import pandas as pd
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def apply_smotified_gan_early_stopping(
    x_train, y_train, x_val, y_val, 
    max_epochs=2000, batch_size=128, patience=10, eval_freq=25
):
    """
    SMOTified-GAN with integrated early stopping based on validation macro F1-score.
    """
    # Lazy imports
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from imblearn.over_sampling import SMOTE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning SMOTified-GAN (Early Stopping) on device: {device}")

    # 1. Minority class identification
    unique, counts = np.unique(y_train, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    
    # 2. Preliminary SMOTE stage
    min_samples = np.min(counts)
    k_neighbors = min(5, max(1, min_samples - 1))
    
    print(f"Step 1: SMOTE (Preliminary synthesis, k={k_neighbors})...")
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    x_smoted, y_smoted = smote.fit_resample(x_train, y_train)

    n_orig = len(x_train)
    x_smote_only = x_smoted[n_orig:]
    n_generated = len(x_smote_only)

    if n_generated == 0:
        print("Data is already balanced. Returning original data.")
        return x_train, y_train

    # 3. Data preparation for GAN training
    minority_data = x_train[y_train == minority_class]
    real_tensor = torch.FloatTensor(minority_data).to(device)
    smote_tensor = torch.FloatTensor(x_smote_only).to(device)

    dataset = TensorDataset(real_tensor)
    drop_last = True if len(dataset) > batch_size else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    input_dim = x_train.shape[1]

    # Network architecture definitions
    def get_generator_block(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True))

    class Generator(nn.Module):
        def __init__(self, in_dim, hid_dim=128):
            super().__init__()
            self.gen = nn.Sequential(
                get_generator_block(in_dim, hid_dim),
                get_generator_block(hid_dim, hid_dim * 2),
                get_generator_block(hid_dim * 2, hid_dim * 4),
                get_generator_block(hid_dim * 4, hid_dim * 8),
                nn.Linear(hid_dim * 8, in_dim),
                nn.Sigmoid()
            )
        def forward(self, x): return self.gen(x)

    def get_discriminator_block(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True))

    class Discriminator(nn.Module):
        def __init__(self, in_dim, hid_dim=128):
            super().__init__()
            self.disc = nn.Sequential(
                get_discriminator_block(in_dim, hid_dim * 4),
                get_discriminator_block(hid_dim * 4, hid_dim * 2),
                get_discriminator_block(hid_dim * 2, hid_dim),
                nn.Linear(hid_dim, 1)
            )
        def forward(self, x): return self.disc(x)

    gen = Generator(input_dim).to(device)
    disc = Discriminator(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    lr = 0.00001
    gen_opt = optim.Adam(gen.parameters(), lr=lr)
    disc_opt = optim.Adam(disc.parameters(), lr=lr)

    # --- Early Stopping Tracking ---
    best_val_f1 = -1.0
    patience_counter = 0
    best_gen_weights = None
    generated_labels = np.full((n_generated,), minority_class)
    
    # Fast evaluation model
    eval_clf = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42, n_jobs=-1)

    print(f"Step 2: Training GAN with early stopping (Max epochs: {max_epochs})...")
    
    for epoch in range(max_epochs):
        gen.train()
        disc.train()
        
        for (real_batch,) in dataloader:
            cur_batch_size = real_batch.size(0)
            real_batch = real_batch.to(device)
            idx = torch.randint(0, len(smote_tensor), (cur_batch_size,), device=device)
            smote_batch_input = smote_tensor[idx]

            # Train Discriminator
            disc_opt.zero_grad()
            disc_real_pred = disc(real_batch)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            fake_refined = gen(smote_batch_input)
            disc_fake_pred = disc(fake_refined.detach())
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            disc_opt.step()

            # Train Generator
            gen_opt.zero_grad()
            fake_refined = gen(smote_batch_input)
            disc_fake_pred = disc(fake_refined)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

        # --- Validation Evaluation every `eval_freq` epochs ---
        if epoch % eval_freq == 0:
            gen.eval()
            with torch.no_grad():
                current_fakes = gen(smote_tensor).cpu().numpy()
            
            x_temp = np.vstack([x_train, current_fakes])
            y_temp = np.hstack([y_train, generated_labels])
            
            eval_clf.fit(x_temp, y_temp)
            val_preds = eval_clf.predict(x_val)
            
            current_f1 = f1_score(y_val, val_preds, average='macro')
            
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_gen_weights = copy.deepcopy(gen.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % (eval_freq * 4) == 0:
                print(f"   Epoch {epoch:04d} | Current F1: {current_f1:.4f} | Best F1: {best_val_f1:.4f} | Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience * eval_freq} epochs. Terminated at epoch {epoch}.")
                break

    # --- Final Generation ---
    if best_gen_weights is not None:
        print(f"Loading best generator checkpoint (Validation F1: {best_val_f1:.4f})...")
        gen.load_state_dict(best_gen_weights)
    else:
        print("Convergence threshold not reached. Utilizing weights from final epoch.")
        
    gen.eval()
    with torch.no_grad():
        final_fakes = gen(smote_tensor).cpu().numpy()

    # Append synthetic samples to training set
    x_balanced = np.vstack([x_train, final_fakes])
    y_balanced = np.hstack([y_train, generated_labels])

    return x_balanced, y_balanced