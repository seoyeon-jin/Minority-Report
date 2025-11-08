"""Time-MMD 데이터셋 로더"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class TimeMMDDataset(Dataset):
    """
    Time-MMD 멀티모달 시계열 데이터셋
    
    Returns:
        xA: (T, dA) - numerical 시계열
        xB: (T, dB) - textual 시계열 (임베딩 또는 통계값)
        maskA: (T,) - numerical 마스크
        maskB: (T,) - textual 마스크
    """
    
    def __init__(self, domain, data_type='numerical', window_size=128, stride=32, 
                 split='train', split_ratio=(0.6, 0.2, 0.2), normalize=True,
                 root_dir='.'):
        """
        Args:
            domain: 'Agriculture', 'Climate', etc.
            data_type: 'numerical' or 'textual'
            window_size: 윈도우 길이
            stride: 슬라이딩 윈도우 stride
            split: 'train', 'val', 'test'
            split_ratio: (train, val, test) 비율
            normalize: 표준화 여부
            root_dir: 데이터 루트 디렉토리
        """
        self.domain = domain
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.normalize = normalize
        
        # 데이터 로드
        self.numerical_data, self.textual_data = self._load_data(root_dir, domain)
        
        # 시간축 정렬
        self.aligned_data = self._align_timeseries()
        
        # Train/Val/Test 분할
        self.split_data = self._split_data(split_ratio)
        
        # 표준화
        if normalize and split == 'train':
            self.scaler_A = StandardScaler()
            self.scaler_B = StandardScaler()
            self._fit_scalers()
        elif normalize:
            # Val/Test는 train의 scaler 사용 (별도로 전달받아야 함)
            self.scaler_A = None
            self.scaler_B = None
        
        # 윈도우 생성
        self.windows = self._create_windows()
    
    def _load_data(self, root_dir, domain):
        """Numerical과 Textual 데이터 로드"""
        # Numerical data
        num_path = Path(root_dir) / 'numerical' / domain / f'{domain}.csv'
        num_df = pd.read_csv(num_path)
        num_df['Date'] = pd.to_datetime(num_df['Date'])
        num_df = num_df.sort_values('Date')
        
        # Textual data (search 사용)
        text_path = Path(root_dir) / 'textual' / domain / f'{domain}_search.csv'
        text_df = pd.read_csv(text_path)
        text_df['start_date'] = pd.to_datetime(text_df['start_date'])
        text_df['end_date'] = pd.to_datetime(text_df['end_date'])  # end_date도 변환!
        text_df = text_df.sort_values('start_date')
        
        return num_df, text_df
    
    def _align_timeseries(self):
        """시간축을 정렬하여 같은 타임스탬프로 매칭"""
        # Numerical: Date 기준
        num_dates = pd.to_datetime(self.numerical_data['Date'])
        
        # Textual: start_date 기준
        text_dates = pd.to_datetime(self.textual_data['start_date'])
        
        # 공통 날짜 범위 찾기
        all_dates = pd.date_range(
            start=max(num_dates.min(), text_dates.min()),
            end=min(num_dates.max(), text_dates.max()),
            freq='MS'  # Month Start
        )
        
        aligned = []
        for date in all_dates:
            # Numerical 데이터
            num_row = self.numerical_data[self.numerical_data['Date'] == date]
            if len(num_row) > 0:
                num_values = num_row.select_dtypes(include=[np.number]).values.flatten()
                mask_A = 1
            else:
                num_values = np.zeros(self._get_num_features())
                mask_A = 0
            
            # Textual 데이터 (fact 길이를 특성으로 사용 - 간단한 예시)
            text_row = self.textual_data[
                (self.textual_data['start_date'] <= date) & 
                (self.textual_data['end_date'] >= date)
            ]
            if len(text_row) > 0:
                # 간단히 fact의 길이를 특성으로 (실제로는 임베딩 사용)
                fact = str(text_row.iloc[0]['fact'])
                text_values = np.array([len(fact), fact.count(' ')])  # 단순 통계
                mask_B = 0 if fact == 'NA' else 1
            else:
                text_values = np.zeros(2)
                mask_B = 0
            
            aligned.append({
                'date': date,
                'xA': num_values,
                'xB': text_values,
                'maskA': mask_A,
                'maskB': mask_B
            })
        
        return aligned
    
    def _get_num_features(self):
        """Numerical feature 개수"""
        return len(self.numerical_data.select_dtypes(include=[np.number]).columns)
    
    def _split_data(self, split_ratio):
        """시간 기준으로 데이터 분할"""
        n = len(self.aligned_data)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])
        
        if self.split == 'train':
            return self.aligned_data[:train_end]
        elif self.split == 'val':
            return self.aligned_data[train_end:val_end]
        else:  # test
            return self.aligned_data[val_end:]
    
    def _fit_scalers(self):
        """Train 데이터로 scaler fit"""
        xA_all = np.array([d['xA'] for d in self.split_data])
        xB_all = np.array([d['xB'] for d in self.split_data])
        
        # 유효한 데이터만 사용
        maskA_all = np.array([d['maskA'] for d in self.split_data])
        maskB_all = np.array([d['maskB'] for d in self.split_data])
        
        if maskA_all.sum() > 0:
            self.scaler_A.fit(xA_all[maskA_all == 1])
        if maskB_all.sum() > 0:
            self.scaler_B.fit(xB_all[maskB_all == 1])
    
    def _create_windows(self):
        """슬라이딩 윈도우 생성"""
        windows = []
        n = len(self.split_data)
        
        for i in range(0, n - self.window_size + 1, self.stride):
            window = self.split_data[i:i + self.window_size]
            windows.append(window)
        
        return windows
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        xA = np.array([d['xA'] for d in window], dtype=np.float32)
        xB = np.array([d['xB'] for d in window], dtype=np.float32)
        maskA = np.array([d['maskA'] for d in window], dtype=np.float32)
        maskB = np.array([d['maskB'] for d in window], dtype=np.float32)
        
        # 표준화
        if self.normalize and self.scaler_A is not None:
            valid_A = maskA == 1
            if valid_A.sum() > 0:
                xA[valid_A] = self.scaler_A.transform(xA[valid_A])
        
        if self.normalize and self.scaler_B is not None:
            valid_B = maskB == 1
            if valid_B.sum() > 0:
                xB[valid_B] = self.scaler_B.transform(xB[valid_B])
        
        return {
            'xA': torch.FloatTensor(xA),
            'xB': torch.FloatTensor(xB),
            'maskA': torch.FloatTensor(maskA),
            'maskB': torch.FloatTensor(maskB)
        }
    
    def get_scalers(self):
        """Scaler 반환 (val/test에서 사용)"""
        return self.scaler_A, self.scaler_B
    
    def set_scalers(self, scaler_A, scaler_B):
        """Scaler 설정"""
        self.scaler_A = scaler_A
        self.scaler_B = scaler_B


def create_dataloaders(domain, batch_size=64, window_size=128, stride=32,
                       split_ratio=(0.6, 0.2, 0.2), normalize=True, 
                       num_workers=0, root_dir='.'):
    """
    Train/Val/Test DataLoader 생성
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Train dataset
    train_dataset = TimeMMDDataset(
        domain=domain,
        window_size=window_size,
        stride=stride,
        split='train',
        split_ratio=split_ratio,
        normalize=normalize,
        root_dir=root_dir
    )
    
    # Val/Test dataset (train의 scaler 사용)
    scaler_A, scaler_B = train_dataset.get_scalers()
    
    val_dataset = TimeMMDDataset(
        domain=domain,
        window_size=window_size,
        stride=stride,
        split='val',
        split_ratio=split_ratio,
        normalize=normalize,
        root_dir=root_dir
    )
    val_dataset.set_scalers(scaler_A, scaler_B)
    
    test_dataset = TimeMMDDataset(
        domain=domain,
        window_size=window_size,
        stride=stride,
        split='test',
        split_ratio=split_ratio,
        normalize=normalize,
        root_dir=root_dir
    )
    test_dataset.set_scalers(scaler_A, scaler_B)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

