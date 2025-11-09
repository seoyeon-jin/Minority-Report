# 🚀 바로 실행 가능한 FOCAL 시각화 데모

## ⚡ 빠른 시작 (10초!)

```bash
cd /Users/sheoyonjin/Desktop/Minority-Report/focal
python demo_visualization.py
```

끝! 🎉 `focal_demo_result.png` 파일이 생성됩니다.

## 📦 필요한 패키지

```bash
pip install torch numpy matplotlib seaborn scikit-learn
```

## 🎨 생성되는 시각화

**한 장의 이미지에 6개 subplot:**

1. **t-SNE Shared/Private** - 모달리티 분리 확인
2. **Similarity Heatmap** - 교차 유사도
3. **Orthogonality 분포** - 직교성 검증
4. **Variance 비율** - 정보 분포
5. **Feature Correlation** - 상관관계
6. **종합 점수** - 0~100점 자동 평가

## 📸 결과 예시

```
focal_demo_result.png (20x12 inch, 150 DPI)

┌─────────────────────┬──────────┐
│  t-SNE              │ Heatmap  │
│  Shared/Private     │          │
│                     │──────────│
│  Combined View      │ Ortho    │
├──────┬──────┬───────┴──────────┤
│Var   │Corr  │  Overall: 82/100 │
│Ratio │      │  ✅ 우수         │
└──────┴──────┴──────────────────┘
```

## 🔧 실제 모델에 적용

### 방법 1: 학습 중 사용

```python
from demo_visualization import FOCALVisualizerDemo

visualizer = FOCALVisualizerDemo()

# 학습 루프에서
for epoch in range(epochs):
    # ... 학습 ...
    
    if epoch % 10 == 0:
        with torch.no_grad():
            # 실제 모델 features 추출
            mod_features, _ = model(batch1, batch2)
            
            # 시각화
            visualizer.visualize_all(
                mod_features,
                f'results/epoch_{epoch:04d}.png'
            )
```

### 방법 2: 체크포인트 분석

```python
from demo_visualization import FOCALVisualizerDemo
import torch

# 모델 로드
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint)
model.eval()

# Features 추출
dataloader = create_dataloader(...)
batch = next(iter(dataloader))

with torch.no_grad():
    mod_features, _ = model(
        batch['aug1'].cuda(),
        batch['aug2'].cuda()
    )

# 시각화
visualizer = FOCALVisualizerDemo()
visualizer.visualize_all(mod_features, 'final_analysis.png')
```

## 📊 판정 기준

### ✅ 우수 (Overall > 70)
- Shared space: 모달리티 잘 섞임
- Private space: 모달리티 잘 분리됨
- Orthogonality: < 0.1
- 색상: 초록색

### ⚠️ 보통 (40~70)
- 부분적으로 분리됨
- 추가 학습 또는 하이퍼파라미터 조정 필요
- 색상: 주황색

### ❌ 불량 (< 40)
- 분리 실패
- Loss weight 재조정 필요
- 색상: 빨강색

## 💡 커스터마이징

```python
# Figure 크기 조정
visualizer = FOCALVisualizerDemo(figsize=(30, 18))

# 색상 변경
visualizer.colors['shared'] = '#your_color'

# 해상도 변경 (savefig 부분 수정)
plt.savefig('result.png', dpi=300)  # 더 높은 해상도
```

## 🐛 문제 해결

### 1. "No module named 'torch'"
```bash
pip install torch
```

### 2. "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### 3. 메모리 부족
```python
# 더미 데이터 크기 줄이기
mod_features = create_dummy_features(
    batch_size=32,  # 64 → 32
    feature_dim=128  # 256 → 128
)
```

## 📝 파일 구조

```
focal/
├── demo_visualization.py      ← 바로 실행!
├── visualize_focal_separation.py  ← 실제 사용
├── VISUALIZATION_GUIDE.md     ← 상세 가이드
└── README_DEMO.md            ← 이 파일
```

## 🎯 다음 단계

1. ✅ `demo_visualization.py` 실행 → 시각화 확인
2. ✅ 결과 해석 방법 학습
3. ✅ 실제 모델에 적용
4. ✅ 학습 과정 모니터링

---

**궁금한 점?**
- 상세 가이드: `VISUALIZATION_GUIDE.md`
- 실제 사용: `visualize_focal_separation.py`

