# 🎨 FOCAL 시각화 가이드

## 📊 생성되는 시각화 (6가지)

### 1. **t-SNE 비교 시각화** (좌상단, 큰 영역)
```
Shared Space        Private Space
[모달리티 섞임]    [모달리티 분리]
     ○ ●               ○○   ●●
    ○ ● ●             ○○     ●●
     ● ○              ○       ●
```
- **Shared**: 모달리티가 섞여야 함 ✓
- **Private**: 모달리티별로 분리되어야 함 ✓

### 2. **Similarity Heatmap** (우상단)
```
         ModA   ModA   ModB   ModB
         Shr    Pri    Shr    Pri
ModA Shr  1.0   0.1    0.7    0.2
ModA Pri  0.1   1.0    0.2    0.3
ModB Shr  0.7   0.2    1.0    0.1
ModB Pri  0.2   0.3    0.1    1.0
```
- **Shared-Shared**: 높음 (빨강) ✓
- **Private-Private**: 낮음 (초록) ✓

### 3. **Orthogonality 분포** (우중단)
```
Frequency
    |     목표 <0.1
    |  ▂▅██▅▂    |
    |▂▅███████▅▂ |
    +─────────────+
       |Cos Sim|
```
- Mean < 0.1이면 잘 분리됨 ✓

### 4. **Variance 비율** (좌하단)
```
   100% ┤ [Private 40%]
        │ [Shared 60%]
        └────────────
        ModA    ModB
```
- 30~70% 범위가 이상적 ✓

### 5. **Feature Correlation** (중하단)
```
Private
   ^
   │ ·  ·
   │  ·  ·
   │ ·    ·
   └────────> Shared
```
- 상관계수 낮을수록 독립적 ✓

### 6. **종합 점수** (우하단)
```
📊 종합 평가

Orthogonality:       ████████░░ 85
Shared Similarity:   ██████░░░░ 72
Private Dissimilar:  ███████░░░ 78
Variance Balance:    ██████████ 95

Overall: 82.5/100
✅ 우수: 잘 분리됨
```

## 🚀 사용 방법

### 방법 1: 학습 루프에 통합

```python
# focal/src/train_utils/pretrain.py 또는 finetune.py

from visualize_focal_separation import FOCALSeparationVisualizer

visualizer = FOCALSeparationVisualizer(figsize=(20, 12))

# 학습 루프에서
for epoch in range(epochs):
    # ... 학습 코드 ...
    
    # 매 10 epoch마다 시각화
    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            batch = next(iter(val_loader))
            mod_features, _ = model(batch['input1'], batch['input2'])
            
            # 시각화 생성
            save_path = f'visualizations/epoch_{epoch:04d}.png'
            visualizer.visualize_all(mod_features, save_path)
            print(f"✓ 시각화 저장: {save_path}")
```

### 방법 2: 학습 완료 후 분석

```python
# 별도 스크립트에서 실행

import torch
from visualize_focal_separation import FOCALSeparationVisualizer

# 1. 체크포인트 로드
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 데이터 로드
dataloader = create_val_dataloader(...)
batch = next(iter(dataloader))

# 3. Forward
with torch.no_grad():
    mod_features, _ = model(
        batch['aug1'].cuda(),
        batch['aug2'].cuda(),
        proj_head=False
    )

# 4. 시각화
visualizer = FOCALSeparationVisualizer()
visualizer.visualize_all(mod_features, 'final_analysis.png')
```

### 방법 3: 여러 체크포인트 비교

```python
import glob
from visualize_focal_separation import FOCALSeparationVisualizer

visualizer = FOCALSeparationVisualizer()
checkpoints = sorted(glob.glob('checkpoints/epoch_*.pth'))

for ckpt_path in checkpoints:
    epoch = int(ckpt_path.split('_')[-1].split('.')[0])
    
    # 로드 & 시각화
    model.load_state_dict(torch.load(ckpt_path))
    mod_features = get_features(model, dataloader)
    
    visualizer.visualize_all(
        mod_features,
        f'comparison/epoch_{epoch:04d}.png'
    )

# GIF 생성 (선택사항)
import imageio
images = [imageio.imread(f) for f in sorted(glob.glob('comparison/*.png'))]
imageio.mimsave('training_progress.gif', images, duration=0.5)
```

## 📈 판정 기준

### ✅ 우수한 분리 (Overall > 70)
```
Orthogonality:     < 0.1
Shared Similarity: > 0.6
Private Dissim:    < 0.3
Variance Balance:  30~70%
```

**시각적 특징:**
- t-SNE Shared: 모달리티 완전히 섞임 🔴🔵 혼합
- t-SNE Private: 모달리티 명확히 분리 🔴 | 🔵
- Heatmap: Shared-Shared 빨강, Private-Private 초록
- Histogram: 0.1 왼쪽에 집중

### ⚠️ 보통 (40 < Overall < 70)
```
Orthogonality:     0.1~0.2
Shared Similarity: 0.4~0.6
Private Dissim:    0.3~0.5
```

**개선 방법:**
- Orthogonality loss weight 증가
- 더 많은 epoch 학습
- Learning rate 조정

### ❌ 불량 (Overall < 40)
```
Orthogonality:     > 0.3
Shared Similarity: < 0.4
Private Dissim:    > 0.5
```

**문제 진단:**
- Shared/Private가 구별 안 됨
- Loss balance 재조정 필요
- 데이터 augmentation 확인

## 🎬 학습 과정 모니터링

### TensorBoard 통합

```python
from torch.utils.tensorboard import SummaryWriter
from visualize_focal_separation import FOCALSeparationVisualizer

writer = SummaryWriter('runs/focal')
visualizer = FOCALSeparationVisualizer()

for epoch in range(epochs):
    # ... 학습 ...
    
    if epoch % 10 == 0:
        # 시각화 생성
        import matplotlib.pyplot as plt
        fig = visualizer.create_figure(mod_features)
        writer.add_figure('Separation/analysis', fig, epoch)
        plt.close(fig)
```

### 실시간 대시보드

```python
# Streamlit 대시보드 (선택사항)
import streamlit as st
from visualize_focal_separation import FOCALSeparationVisualizer

st.title("FOCAL 분리 품질 모니터링")

epoch = st.slider("Epoch", 0, 1000, 100)
checkpoint = f"checkpoints/epoch_{epoch}.pth"

if st.button("분석"):
    # 로드 & 시각화
    visualizer = FOCALSeparationVisualizer()
    visualizer.visualize_all(mod_features, 'temp.png')
    st.image('temp.png')
```

## 💡 팁

### 1. 메모리 절약
```python
# 전체 validation set 대신 샘플만 사용
sample_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
batch = next(iter(sample_loader))
```

### 2. 고해상도 저장
```python
visualizer = FOCALSeparationVisualizer(figsize=(30, 18))  # 더 큰 figure
visualizer.visualize_all(mod_features, 'high_res.png')
# 또는 PDF로
# plt.savefig('analysis.pdf', format='pdf')
```

### 3. 배치 비교
```python
# 여러 배치 평균
all_features = []
for batch in val_loader:
    features = model(batch['input1'], batch['input2'])[0]
    all_features.append(features)

# 평균 features로 시각화
avg_features = {
    mod: torch.stack([f[mod] for f in all_features]).mean(0)
    for mod in all_features[0].keys()
}
visualizer.visualize_all(avg_features, 'averaged.png')
```

## 📸 예시 출력

```
focal_visualizations/
├── epoch_0000.png      ← 초기 (랜덤)
├── epoch_0100.png      ← 학습 중
├── epoch_0500.png      ← 수렴 중
├── epoch_1000.png      ← 최종
└── final_analysis.png  ← Best checkpoint
```

각 이미지는 6개 subplot으로 구성:
- 20 x 12 inch (기본)
- 150 DPI
- PNG 형식

## 🎯 빠른 체크리스트

학습이 끝난 후 이것만 확인하세요:

```bash
# 1. 시각화 생성
python -c "from visualize_focal_separation import *; \
           visualize_from_checkpoint('best.pth', dataloader)"

# 2. 결과 확인
open focal_visualizations/analysis.png

# 3. 체크리스트
# □ t-SNE Shared: 모달리티 섞임? 
# □ t-SNE Private: 모달리티 분리?
# □ Heatmap: Shared-Shared 빨강?
# □ Histogram: <0.1 영역에 집중?
# □ Variance: 30~70% 범위?
# □ Overall Score: >70?
```

모두 ✓ 이면 성공! 🎉

