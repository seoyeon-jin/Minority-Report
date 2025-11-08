# 빠른 시작 가이드

## 📋 체크리스트

✅ 프로젝트 구조 완성!

```
✓ data/              # 원시/전처리 데이터
✓ src/
  ✓ datamod/         # 데이터로더·전처리
  ✓ metrics/         # 메트릭 함수 (MAE, DTW, Pearson r)
  ✓ baselines/       # CCA+Linear, Cross-AE
  ✓ utils/           # 공통 유틸 (시드, 로깅)
  ✓ train.py         # 통합 학습/평가 스크립트
✓ configs/           # 하이퍼파라미터 (YAML)
✓ reports/           # 결과 저장 위치
✓ viz/               # 시각화 저장 위치
```

## 🚀 5분 안에 시작하기

### 1단계: 패키지 설치 (1분)

```bash
cd /Users/sheoyonjin/Desktop/Minority-Report
pip install -r requirements.txt
```

### 2단계: 파이프라인 테스트 (1분)

```bash
python test_pipeline.py
```

이 명령어는 다음을 확인합니다:
- ✓ 데이터 로딩 (xA, xB, maskA, maskB 형태)
- ✓ Train/Val/Test 배치 순회
- ✓ 메트릭 계산 (MAE, DTW, Pearson r)

### 3단계: 첫 번째 실험 (3분)

```bash
# CCA+Linear 베이스라인 실행
python src/train.py --model cca --domain Agriculture

# 결과 확인
cat reports/results_Agriculture_cca.csv
```

## 📊 지원하는 도메인

- `Agriculture` - 농업
- `Climate` - 기후
- `Economy` - 경제
- `Energy` - 에너지
- `Environment` - 환경
- `Health_AFR` - 아프리카 건강
- `Health_US` - 미국 건강
- `Security` - 안보
- `SocialGood` - 사회선
- `Traffic` - 교통

## 🎯 실험 시나리오

### 시나리오 1: 빠른 베이스라인 비교

```bash
# 모든 베이스라인 한번에 (CCA+Linear & Cross-AE)
python src/train.py --domain Agriculture --model all
```

### 시나리오 2: 여러 도메인 실험

```bash
# 스크립트 작성
for domain in Agriculture Climate Economy; do
    python src/train.py --domain $domain --model all
done
```

### 시나리오 3: 여러 시드로 안정성 확인

```bash
python src/train.py --domain Agriculture --seeds 42 123 2025 456 789
```

### 시나리오 4: 하이퍼파라미터 튜닝

1. `configs/default.yaml` 복사:
```bash
cp configs/default.yaml configs/exp1.yaml
```

2. `configs/exp1.yaml` 수정 (예: window_T를 256으로):
```yaml
data:
  window_T: 256
  stride: 64
```

3. 실행:
```bash
python src/train.py --config configs/exp1.yaml --domain Agriculture
```

## 📈 결과 확인

결과는 `reports/` 디렉토리에 저장됩니다:

```bash
ls -lh reports/
# results_Agriculture_cca.csv
# results_Agriculture_cross_ae.csv
# results_Agriculture_all.csv
```

CSV 형식:
```csv
model,domain,seed,A2B_mae,A2B_dtw,A2B_pearson,B2A_mae,B2A_dtw,B2A_pearson
CCA+Linear,Agriculture,42,0.523,12.34,0.765,0.612,15.23,0.701
```

### Python으로 결과 분석

```python
import pandas as pd

# 결과 로드
df = pd.read_csv('reports/results_Agriculture_all.csv')

# 평균 성능
print(df.groupby('model')[['A2B_mae', 'B2A_mae', 'A2B_pearson', 'B2A_pearson']].mean())

# 최고 성능 모델
best_model = df.loc[df['A2B_mae'].idxmin()]
print(f"Best model: {best_model['model']} (MAE: {best_model['A2B_mae']:.4f})")
```

## ⚙️ 주요 설정 (configs/default.yaml)

```yaml
data:
  window_T: 128        # 윈도우 길이 (늘리면 더 긴 패턴 포착)
  stride: 32           # 슬라이딩 간격 (줄이면 더 많은 샘플)
  split_ratio:         # Train/Val/Test 비율
    train: 0.6
    val: 0.2
    test: 0.2

train:
  batch_size: 64       # 메모리 부족시 줄이기
  epochs: 30           # Cross-AE 학습 epoch 수
  lr: 0.0003           # 학습률
  early_stopping_patience: 10

model:
  cca_components: 10   # CCA 컴포넌트 수
  ae_hidden_dim: 64    # AutoEncoder hidden 차원
  ae_latent_dim: 32    # Latent 차원
```

## 🔧 문제 해결

### 문제 1: 메모리 부족

```yaml
# configs/default.yaml 수정
train:
  batch_size: 16    # 64 -> 16
data:
  window_T: 64      # 128 -> 64
```

### 문제 2: 데이터 로드 오류

```bash
# Time-MMD 폴더 확인
ls Time-MMD/numerical/Agriculture/
# Agriculture.csv가 있어야 함
```

### 문제 3: GPU 메모리 부족

Cross-AE는 자동으로 CPU를 사용합니다. 또는:

```yaml
# configs/default.yaml 수정
model:
  ae_hidden_dim: 32   # 64 -> 32
  ae_latent_dim: 16   # 32 -> 16
```

## 📝 종료 기준 확인

요구사항의 3개 체크리스트:

```bash
# 1. train.py 실행 → train/val/test 배치 순회
python test_pipeline.py
# 출력: "✓ Train/Val/Test 배치 순회 OK"

# 2. MAE/DTW/r 숫자 저장되고 로그에 보임
python src/train.py --model cca --domain Agriculture
# 출력: "A->B: MAE=0.xxx, DTW=xx.xx, Pearson=0.xxx"

# 3. CCA+Linear & Cross-AE 결과 CSV 생성
ls reports/
# results_Agriculture_all.csv 확인
```

## 🎓 다음 단계

1. **다양한 도메인 실험**
   ```bash
   for domain in Agriculture Climate Economy Energy; do
       python src/train.py --domain $domain --model all
   done
   ```

2. **결과 시각화**
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   # 모든 결과 로드
   results = []
   for domain in ['Agriculture', 'Climate', 'Economy']:
       df = pd.read_csv(f'reports/results_{domain}_all.csv')
       df['domain'] = domain
       results.append(df)
   
   all_results = pd.concat(results)
   
   # 도메인별 MAE 비교
   all_results.groupby('domain')['A2B_mae'].mean().plot(kind='bar')
   plt.savefig('viz/mae_by_domain.png')
   ```

3. **새로운 모델 추가**
   - `src/baselines/` 폴더에 새 모델 추가
   - `src/train.py`에 통합

4. **논문 작성용 표 생성**
   ```python
   import pandas as pd
   
   df = pd.read_csv('reports/results_Agriculture_all.csv')
   
   # LaTeX 표 생성
   print(df.groupby('model').mean().to_latex(float_format="%.3f"))
   ```

## 🎉 완료!

이제 연구를 시작할 준비가 되었습니다!

질문이나 문제가 있으면 `PROJECT_README.md`를 참고하세요.

