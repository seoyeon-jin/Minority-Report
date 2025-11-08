"""CCA + Linear Regression 베이스라인"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge


class CCALinearBaseline:
    """
    CCA로 공통 표현 학습 후 선형 회귀로 교차 복원
    """
    
    def __init__(self, n_components=10, alpha=1.0):
        """
        Args:
            n_components: CCA 컴포넌트 수
            alpha: Ridge regression 정규화 파라미터
        """
        self.n_components = n_components
        self.alpha = alpha
        
        self.cca = CCA(n_components=n_components)
        self.regressor_A2B = Ridge(alpha=alpha)
        self.regressor_B2A = Ridge(alpha=alpha)
        
        self.fitted = False
    
    def fit(self, train_loader):
        """
        Train 데이터로 CCA와 회귀 모델 학습
        
        Args:
            train_loader: DataLoader
        """
        # 데이터 수집
        xA_list, xB_list = [], []
        maskA_list, maskB_list = [], []
        
        for batch in train_loader:
            xA = batch['xA']  # (B, T, dA)
            xB = batch['xB']  # (B, T, dB)
            maskA = batch['maskA']  # (B, T)
            maskB = batch['maskB']  # (B, T)
            
            # Flatten: (B*T, d)
            B, T = xA.shape[0], xA.shape[1]
            xA_flat = xA.reshape(-1, xA.shape[-1])  # (B*T, dA)
            xB_flat = xB.reshape(-1, xB.shape[-1])  # (B*T, dB)
            maskA_flat = maskA.reshape(-1)  # (B*T,)
            maskB_flat = maskB.reshape(-1)  # (B*T,)
            
            xA_list.append(xA_flat.numpy())
            xB_list.append(xB_flat.numpy())
            maskA_list.append(maskA_flat.numpy())
            maskB_list.append(maskB_flat.numpy())
        
        xA_all = np.concatenate(xA_list, axis=0)
        xB_all = np.concatenate(xB_list, axis=0)
        maskA_all = np.concatenate(maskA_list, axis=0)
        maskB_all = np.concatenate(maskB_list, axis=0)
        
        # 유효한 샘플만 사용 (둘 다 valid)
        valid_mask = (maskA_all == 1) & (maskB_all == 1)
        xA_valid = xA_all[valid_mask]
        xB_valid = xB_all[valid_mask]
        
        if len(xA_valid) < self.n_components:
            print(f"Warning: Not enough valid samples ({len(xA_valid)}). Using all data.")
            xA_valid = xA_all
            xB_valid = xB_all
        
        # CCA fit
        print(f"Fitting CCA with {len(xA_valid)} samples...")
        self.cca.fit(xA_valid, xB_valid)
        
        # CCA transform
        xA_cca, xB_cca = self.cca.transform(xA_valid, xB_valid)
        
        # Linear regression: CCA_A -> xB
        print("Fitting A->B regressor...")
        self.regressor_A2B.fit(xA_cca, xB_valid)
        
        # Linear regression: CCA_B -> xA
        print("Fitting B->A regressor...")
        self.regressor_B2A.fit(xB_cca, xA_valid)
        
        self.fitted = True
        print("CCA+Linear model fitted!")
    
    def predict_B_from_A(self, xA):
        """A로부터 B 예측"""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet!")
        
        if isinstance(xA, torch.Tensor):
            xA = xA.detach().cpu().numpy()
        
        original_shape = xA.shape
        if len(xA.shape) == 3:
            B, T, dA = xA.shape
            xA = xA.reshape(-1, dA)
        
        # CCA transform
        xA_cca = self.cca.transform(xA, None)[0]
        
        # Predict xB
        xB_pred = self.regressor_A2B.predict(xA_cca)
        
        # Reshape back
        if len(original_shape) == 3:
            xB_pred = xB_pred.reshape(B, T, -1)
        
        return xB_pred
    
    def predict_A_from_B(self, xB):
        """B로부터 A 예측"""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet!")
        
        if isinstance(xB, torch.Tensor):
            xB = xB.detach().cpu().numpy()
        
        original_shape = xB.shape
        if len(xB.shape) == 3:
            B, T, dB = xB.shape
            xB = xB.reshape(-1, dB)
        
        # CCA transform
        xB_cca = self.cca.transform(None, xB)[1]
        
        # Predict xA
        xA_pred = self.regressor_B2A.predict(xB_cca)
        
        # Reshape back
        if len(original_shape) == 3:
            xA_pred = xA_pred.reshape(B, T, -1)
        
        return xA_pred
    
    def evaluate(self, test_loader, metrics_fn):
        """
        Test 데이터로 평가
        
        Args:
            test_loader: DataLoader
            metrics_fn: 메트릭 계산 함수
        
        Returns:
            dict: 평가 결과
        """
        results = {
            'A2B': {'mae': [], 'dtw': [], 'pearson': []},
            'B2A': {'mae': [], 'dtw': [], 'pearson': []}
        }
        
        for batch in test_loader:
            xA = batch['xA']
            xB = batch['xB']
            maskA = batch['maskA']
            maskB = batch['maskB']
            
            # A -> B
            xB_pred = self.predict_B_from_A(xA)
            metrics_A2B = metrics_fn(xB_pred, xB, maskB)
            
            # B -> A
            xA_pred = self.predict_A_from_B(xB)
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

