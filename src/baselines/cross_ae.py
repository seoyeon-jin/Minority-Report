"""Cross-AutoEncoder 베이스라인"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Encoder(nn.Module):
    """간단한 Encoder"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        # x: (B, T, D) -> (B, T, latent)
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*T, D)
        z = self.net(x_flat)  # (B*T, latent)
        z = z.reshape(B, T, -1)  # (B, T, latent)
        return z


class Decoder(nn.Module):
    """간단한 Decoder"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        # z: (B, T, latent) -> (B, T, D)
        B, T, latent = z.shape
        z_flat = z.reshape(-1, latent)  # (B*T, latent)
        x = self.net(z_flat)  # (B*T, D)
        x = x.reshape(B, T, -1)  # (B, T, D)
        return x


class CrossAutoEncoder(nn.Module):
    """
    Cross-AutoEncoder for multimodal reconstruction
    E_A, D_B: A -> B
    E_B, D_A: B -> A
    """
    
    def __init__(self, dim_A, dim_B, hidden_dim=64, latent_dim=32):
        super().__init__()
        
        self.encoder_A = Encoder(dim_A, hidden_dim, latent_dim)
        self.encoder_B = Encoder(dim_B, hidden_dim, latent_dim)
        self.decoder_A = Decoder(latent_dim, hidden_dim, dim_A)
        self.decoder_B = Decoder(latent_dim, hidden_dim, dim_B)
    
    def forward(self, xA, xB):
        """
        Args:
            xA: (B, T, dA)
            xB: (B, T, dB)
        
        Returns:
            xA_recon: A->A reconstruction
            xB_recon: B->B reconstruction
            xB_cross: A->B cross-reconstruction
            xA_cross: B->A cross-reconstruction
        """
        # Encode
        zA = self.encoder_A(xA)
        zB = self.encoder_B(xB)
        
        # Decode (reconstruction)
        xA_recon = self.decoder_A(zA)
        xB_recon = self.decoder_B(zB)
        
        # Cross-decode
        xB_cross = self.decoder_B(zA)  # A -> B
        xA_cross = self.decoder_A(zB)  # B -> A
        
        return xA_recon, xB_recon, xB_cross, xA_cross
    
    def predict_B_from_A(self, xA):
        """A -> B 예측"""
        with torch.no_grad():
            zA = self.encoder_A(xA)
            xB_pred = self.decoder_B(zA)
        return xB_pred
    
    def predict_A_from_B(self, xB):
        """B -> A 예측"""
        with torch.no_grad():
            zB = self.encoder_B(xB)
            xA_pred = self.decoder_A(zB)
        return xA_pred


class CrossAETrainer:
    """Cross-AE 학습 클래스"""
    
    def __init__(self, model, device='cpu', lr=3e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss(reduction='none')
    
    def train_epoch(self, train_loader):
        """한 epoch 학습"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            xA = batch['xA'].to(self.device)
            xB = batch['xB'].to(self.device)
            maskA = batch['maskA'].to(self.device)
            maskB = batch['maskB'].to(self.device)
            
            # Forward
            xA_recon, xB_recon, xB_cross, xA_cross = self.model(xA, xB)
            
            # Loss (masked)
            # Reconstruction loss
            loss_A_recon = self._masked_loss(xA_recon, xA, maskA)
            loss_B_recon = self._masked_loss(xB_recon, xB, maskB)
            
            # Cross-reconstruction loss
            loss_A2B = self._masked_loss(xB_cross, xB, maskB)
            loss_B2A = self._masked_loss(xA_cross, xA, maskA)
            
            # Total loss
            loss = loss_A_recon + loss_B_recon + loss_A2B + loss_B2A
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _masked_loss(self, pred, target, mask):
        """Masked MSE loss"""
        loss = self.criterion(pred, target)  # (B, T, D)
        
        # Expand mask
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)  # (B, T, D)
        
        # Apply mask
        masked_loss = loss * mask_expanded
        
        # Average
        if mask_expanded.sum() > 0:
            return masked_loss.sum() / mask_expanded.sum()
        else:
            return masked_loss.sum()
    
    def evaluate(self, test_loader, metrics_fn):
        """평가"""
        self.model.eval()
        
        results = {
            'A2B': {'mae': [], 'dtw': [], 'pearson': []},
            'B2A': {'mae': [], 'dtw': [], 'pearson': []}
        }
        
        with torch.no_grad():
            for batch in test_loader:
                xA = batch['xA'].to(self.device)
                xB = batch['xB'].to(self.device)
                maskA = batch['maskA']
                maskB = batch['maskB']
                
                # A -> B
                xB_pred = self.model.predict_B_from_A(xA)
                metrics_A2B = metrics_fn(xB_pred, xB, maskB)
                
                # B -> A
                xA_pred = self.model.predict_A_from_B(xB)
                metrics_B2A = metrics_fn(xA_pred, xA, maskA)
                
                # Collect
                for key in ['mae', 'dtw', 'pearson']:
                    if not np.isnan(metrics_A2B[key]):
                        results['A2B'][key].append(metrics_A2B[key])
                    if not np.isnan(metrics_B2A[key]):
                        results['B2A'][key].append(metrics_B2A[key])
        
        # Average
        for direction in ['A2B', 'B2A']:
            for key in ['mae', 'dtw', 'pearson']:
                if len(results[direction][key]) > 0:
                    results[direction][key] = np.mean(results[direction][key])
                else:
                    results[direction][key] = np.nan
        
        return results
    
    def fit(self, train_loader, val_loader, epochs=30, patience=10):
        """전체 학습 루프"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    xA = batch['xA'].to(self.device)
                    xB = batch['xB'].to(self.device)
                    maskA = batch['maskA'].to(self.device)
                    maskB = batch['maskB'].to(self.device)
                    
                    xA_recon, xB_recon, xB_cross, xA_cross = self.model(xA, xB)
                    
                    loss_A_recon = self._masked_loss(xA_recon, xA, maskA)
                    loss_B_recon = self._masked_loss(xB_recon, xB, maskB)
                    loss_A2B = self._masked_loss(xB_cross, xB, maskB)
                    loss_B2A = self._masked_loss(xA_cross, xA, maskA)
                    
                    loss = loss_A_recon + loss_B_recon + loss_A2B + loss_B2A
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

