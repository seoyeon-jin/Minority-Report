"""Time-MMD 데이터셋 로더 (개선 버전 - 텍스트 정보 포함)"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class TimeMMDDatasetV2(Dataset):
    """
    Time-MMD 멀티모달 시계열 데이터셋 (개선 버전)
    
    Returns:
        xA: (T, dA) - numerical 시계열
        xB: (T, dB) - textual 시계열 (임베딩 또는 통계값)
        maskA: (T,) - numerical 마스크
        maskB: (T,) - textual 마스크
        dates: List[datetime] - 각 시점의 날짜
        texts: List[str] - 각 시점의 원본 텍스트
        numerical_raw: (T, dA) - 표준화 전 numerical 값
    """
    
    def __init__(self, domain, window_size=128, stride=32, 
                 split='train', split_ratio=(0.6, 0.2, 0.2), normalize=True,
                 root_dir='.', text_mode='simple', return_metadata=True):
        """
        Args:
            domain: 'Agriculture', 'Climate', etc.
            window_size: 윈도우 길이
            stride: 슬라이딩 윈도우 stride
            split: 'train', 'val', 'test'
            split_ratio: (train, val, test) 비율
            normalize: 표준화 여부
            root_dir: 데이터 루트 디렉토리
            text_mode: 'simple' (길이/단어수), 'bert', 'sentence-transformer'
            return_metadata: dates, texts 등 메타데이터 반환 여부
        """
        self.domain = domain
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.normalize = normalize
        self.text_mode = text_mode
        self.return_metadata = return_metadata
        
        # Text encoder 초기화 (필요시)
        self._init_text_encoder()
        
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
            self.scaler_A = None
            self.scaler_B = None
        
        # 윈도우 생성
        self.windows = self._create_windows()
    
    def _init_text_encoder(self):
        """Text encoder 초기화"""
        if self.text_mode == 'bert':
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.text_model = AutoModel.from_pretrained('bert-base-uncased')
                self.text_model.eval()
                print("✓ BERT encoder loaded")
            except:
                print("⚠ BERT not available, falling back to simple mode")
                self.text_mode = 'simple'
        
        elif self.text_mode == 'sentence-transformer':
            try:
                from sentence_transformers import SentenceTransformer
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Sentence-Transformer encoder loaded")
            except:
                print("⚠ Sentence-Transformer not available, falling back to simple mode")
                self.text_mode = 'simple'
    
    def _process_text(self, text):
        """텍스트를 벡터로 변환"""
        if self.text_mode == 'simple':
            # 단순 통계: 길이, 단어 수, 문장 수
            return np.array([
                len(text), 
                text.count(' '),
                text.count('.') + text.count('!') + text.count('?')
            ])
        
        elif self.text_mode == 'bert':
            # BERT embedding
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors='pt', 
                                       truncation=True, max_length=512)
                outputs = self.text_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding
        
        elif self.text_mode == 'sentence-transformer':
            # Sentence embedding
            return self.text_model.encode(text)
    
    def _load_data(self, root_dir, domain):
        """Numerical과 Textual 데이터 로드"""
        # Numerical data
        num_path = Path(root_dir) / 'numerical' / domain / f'{domain}.csv'
        num_df = pd.read_csv(num_path)
        
        # 날짜 컬럼 자동 감지 (date 우선, 없으면 다른 컬럼)
        date_col = self._find_date_column(num_df)
        num_df['date'] = pd.to_datetime(num_df[date_col])
        num_df = num_df.sort_values('date')
        
        # Textual data (search 사용)
        text_path = Path(root_dir) / 'textual' / domain / f'{domain}_search.csv'
        text_df = pd.read_csv(text_path)
        text_df['start_date'] = pd.to_datetime(text_df['start_date'])
        text_df['end_date'] = pd.to_datetime(text_df['end_date'])
        text_df = text_df.sort_values('start_date')
        
        return num_df, text_df
    
    def _find_date_column(self, df):
        """날짜 컬럼 자동 찾기"""
        # 우선순위: date > Date > Month > MapDate > 첫 번째 컬럼
        possible_date_cols = ['date', 'Date', 'Month', 'MapDate']
        
        for col in possible_date_cols:
            if col in df.columns:
                return col
        
        # 못 찾으면 첫 번째 컬럼 사용
        print(f"⚠ Warning: No standard date column found. Using first column: {df.columns[0]}")
        return df.columns[0]
    
    def _align_timeseries(self):
        """시간축을 정렬하여 같은 타임스탬프로 매칭"""
        num_dates = pd.to_datetime(self.numerical_data['date'])
        text_dates = pd.to_datetime(self.textual_data['start_date'])
        
        all_dates = pd.date_range(
            start=max(num_dates.min(), text_dates.min()),
            end=min(num_dates.max(), text_dates.max()),
            freq='MS'
        )
        
        aligned = []
        for date in all_dates:
            # Numerical 데이터
            num_row = self.numerical_data[self.numerical_data['date'] == date]
            if len(num_row) > 0:
                num_values = num_row.select_dtypes(include=[np.number]).values.flatten()
                mask_A = 1
            else:
                num_values = np.zeros(self._get_num_features())
                mask_A = 0
            
            # Textual 데이터
            text_row = self.textual_data[
                (self.textual_data['start_date'] <= date) & 
                (self.textual_data['end_date'] >= date)
            ]
            
            if len(text_row) > 0:
                fact = str(text_row.iloc[0]['fact'])
                # 'nan', 'NA', 빈 문자열 처리
                if fact in ['nan', 'NA', '', 'None']:
                    text_values = self._process_text("")
                    mask_B = 0
                    raw_text = "[NO TEXT]"
                else:
                    text_values = self._process_text(fact)
                    mask_B = 1
                    raw_text = fact
            else:
                text_values = self._process_text("")
                mask_B = 0
                raw_text = "[NO TEXT]"
            
            aligned.append({
                'date': date,
                'xA': num_values,
                'xB': text_values,
                'maskA': mask_A,
                'maskB': mask_B,
                'text': raw_text  # ✨ 원본 텍스트 저장
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
        else:
            return self.aligned_data[val_end:]
    
    def _fit_scalers(self):
        """Train 데이터로 scaler fit"""
        xA_all = np.array([d['xA'] for d in self.split_data])
        xB_all = np.array([d['xB'] for d in self.split_data])
        
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
        
        # 원본 저장 (표준화 전)
        xA_raw = xA.copy()
        
        # 표준화
        if self.normalize and self.scaler_A is not None:
            valid_A = maskA == 1
            if valid_A.sum() > 0:
                xA[valid_A] = self.scaler_A.transform(xA[valid_A])
        
        if self.normalize and self.scaler_B is not None:
            valid_B = maskB == 1
            if valid_B.sum() > 0:
                xB[valid_B] = self.scaler_B.transform(xB[valid_B])
        
        result = {
            'xA': torch.FloatTensor(xA),
            'xB': torch.FloatTensor(xB),
            'maskA': torch.FloatTensor(maskA),
            'maskB': torch.FloatTensor(maskB),
        }
        
        # 메타데이터 추가
        if self.return_metadata:
            # pandas Timestamp를 문자열로 변환 (DataLoader 호환)
            result['dates'] = [str(d['date']) for d in window]
            result['texts'] = [d['text'] for d in window]
            result['xA_raw'] = torch.FloatTensor(xA_raw)
        
        return result
    
    def verify_alignment(self, idx=0, n_steps=5):
        """데이터 정렬 확인 (디버깅용)"""
        batch = self[idx]
        
        print("\n" + "=" * 80)
        print(f"📊 Sample {idx} Alignment Verification")
        print("=" * 80)
        
        n_steps = min(n_steps, len(batch['dates']))
        
        for t in range(n_steps):
            print(f"\n[⏰ Time Step {t}]")
            print(f"  📅 Date: {batch['dates'][t]}")
            print(f"  📝 Text: {batch['texts'][t][:150]}..." if len(batch['texts'][t]) > 150 else f"  📝 Text: {batch['texts'][t]}")
            print(f"  🔢 Numerical (first 3): {batch['xA_raw'][t][:3]}")
            print(f"  📊 Text features: {batch['xB'][t][:5]}...")
            print(f"  ✅ Masks: Numerical={batch['maskA'][t].item()}, Text={batch['maskB'][t].item()}")
        
        print("\n" + "=" * 80)
    
    def get_scalers(self):
        """Scaler 반환"""
        return self.scaler_A, self.scaler_B
    
    def set_scalers(self, scaler_A, scaler_B):
        """Scaler 설정"""
        self.scaler_A = scaler_A
        self.scaler_B = scaler_B

