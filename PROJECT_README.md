# Time-MMD 멀티모달 시계열 실험 프레임워크

Time-MMD 데이터셋을 사용한 멀티모달 시계열 분석 실험 프레임워크입니다.

## 프로젝트 구조

```
project/
  ├── data/              # 원시/전처리 데이터
  ├── src/
  │   ├── datamod/      # 데이터로더·전처리
  │   ├── metrics/      # 메트릭 함수 (MAE, DTW, Pearson)
  │   ├── baselines/    # CCA+Linear, Cross-AE
  │   ├── utils/        # 공통 유틸 (시드, 로깅)
  │   └── train.py      # 통합 학습/평가 스크립트
  ├── configs/          # 하이퍼파라미터 (YAML)
  ├── reports/          # 결과 표/그림
  ├── viz/              # 시각화 이미지
  └── Time-MMD/         # 원본 데이터
```

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 형태

모든 데이터는 다음 형태로 통일됩니다:
- `xA`: (B, T, dA) - Numerical 시계열
- `xB`: (B, T, dB) - Textual 시계열 (임베딩/통계)
- `maskA`: (B, T) - Numerical 마스크 (1=유효, 0=결측)
- `maskB`: (B, T) - Textual 마스크 (1=유효, 0=결측)

## 사용법

### 1. 기본 실행 (모든 베이스라인)

```bash
python src/train.py --domain Agriculture
```

### 2. 특정 모델만 실행

```bash
# CCA+Linear만
python src/train.py --model cca --domain Agriculture

# Cross-AE만
python src/train.py --model cross_ae --domain Agriculture
```

### 3. 다른 도메인 실험

```bash
python src/train.py --domain Climate
python src/train.py --domain Economy
python src/train.py --domain Energy
```

### 4. 커스텀 시드 사용

```bash
python src/train.py --seeds 42 123 2025 --domain Agriculture
```

### 5. 커스텀 설정 파일 사용

```bash
python src/train.py --config configs/custom.yaml --domain Agriculture
```

## 설정 파일 (configs/default.yaml)

```yaml
data:
  resample: "M"         # 리샘플링 주기
  window_T: 128         # 윈도우 길이
  stride: 32            # 슬라이딩 stride
  split: "time"         # 분할 방식
  split_ratio:
    train: 0.6
    val: 0.2
    test: 0.2
  normalize: true       # 표준화 여부

train:
  batch_size: 64
  epochs: 30
  lr: 0.0003
  weight_decay: 0.0001
  early_stopping_patience: 10

eval:
  metrics: ["mae", "dtw", "pearson"]

model:
  cca_components: 10
  ae_hidden_dim: 64
  ae_latent_dim: 32

seeds: [42, 123, 2025]
```

## 평가 메트릭

- **MAE** (Mean Absolute Error): 평균 절대 오차
- **DTW** (Dynamic Time Warping): 동적 시간 정렬 거리
- **Pearson r**: 피어슨 상관계수

모든 메트릭은 마스크를 고려하여 **유효한 위치에서만** 계산됩니다.

## 베이스라인 모델

### 1. CCA + Linear Regression
- CCA로 공통 표현 학습
- 선형 회귀로 교차 복원 (A↔B)
- 빠르고 간단한 베이스라인

### 2. Cross-AutoEncoder
- Encoder/Decoder 구조
- E_A + D_B: A→B 교차 복원
- E_B + D_A: B→A 교차 복원
- 재구성 손실 + 교차 복원 손실

## 결과

결과는 `reports/` 디렉토리에 CSV 형태로 저장됩니다:

```
reports/results_Agriculture_all.csv
```

형식:
```
model,domain,seed,A2B_mae,A2B_dtw,A2B_pearson,B2A_mae,B2A_dtw,B2A_pearson
CCA+Linear,Agriculture,42,0.523,12.34,0.765,0.612,15.23,0.701
Cross-AE,Agriculture,42,0.489,11.56,0.798,0.571,14.12,0.734
...
```

## 종료 기준 체크리스트

- [x] 프로젝트 폴더 구조 생성
- [x] 데이터 로더 구현 (xA, xB, maskA, maskB)
- [x] 메트릭 함수 구현 (MAE, DTW, Pearson)
- [x] CCA+Linear 베이스라인 구현
- [x] Cross-AutoEncoder 베이스라인 구현
- [x] 공통 유틸 구현 (시드, 로깅, 설정)
- [x] YAML 설정 파일 생성
- [x] 통합 학습 스크립트 (train.py)
- [ ] 전체 파이프라인 테스트

## 다음 단계

1. 전체 파이프라인 실행 및 검증
2. 다양한 도메인에서 실험
3. 결과 분석 및 시각화
4. 새로운 모델 추가

## 문제 해결

### 데이터 로드 오류
- Time-MMD 폴더가 프로젝트 루트에 있는지 확인
- CSV 파일 경로가 올바른지 확인

### 메모리 부족
- `batch_size`를 줄이기 (configs/default.yaml)
- `window_T`를 줄이기

### GPU 메모리 부족
- Cross-AE는 CPU에서도 실행 가능
- `hidden_dim`, `latent_dim` 줄이기

## 라이선스

MIT License

