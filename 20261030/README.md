# Day 10. Optimizer & Regularization 정리

> 딥러닝 핵심 아키텍처 및 학습 · 2026.10.30
> 주제: Optimizer, Learning Rate, Batch, Regularization, Validation, Early Stopping

## 목차

1. [한눈에 보는 큰 그림](#한눈에-보는-큰-그림)
2. [학습의 기본 흐름](#1-학습의-기본-흐름)
3. [Optimizer, Learning Rate, Scheduler](#2-optimizer-learning-rate-scheduler)
4. [Batch · Iteration · Epoch](#3-batch--iteration--epoch)
5. [Train · Validation · Test](#4-train--validation--test)
6. [Overfitting vs Underfitting](#5-overfitting-vs-underfitting)
7. [Regularization (규제)](#6-regularization-규제)
8. [train() · eval() · no_grad()](#7-train--eval--no_grad)
9. [Early Stopping (조기 종료)](#8-early-stopping-조기-종료)
10. [실험·평가 용어](#9-실험평가-용어)
11. [전체 학습 흐름](#10-전체-학습-흐름)
12. [복습용 요약](#11-복습용-요약)

---

## 한눈에 보는 큰 그림

이 단원은 세 가지 질문으로 요약된다.

| 질문 | 관련 개념 |
|---|---|
| 모델을 **어떻게** 학습시킬까? | Optimizer, Learning Rate, Batch |
| 왜 **새로운 데이터**에서 성능이 떨어질까? | Overfitting, Regularization |
| **어떤 모델**을 최종으로 고를까? | Validation, Early Stopping |

---

## 1. 학습의 기본 흐름

ANN은 예측하고, 정답과의 차이를 계산한 뒤, 그 차이를 줄이는 방향으로 가중치를 고친다.

1. **Forward (순전파)**: 입력으로 예측값 계산
2. **Loss**: 예측과 정답의 차이 계산
3. **Backward (역전파)**: 각 파라미터의 기울기(Gradient) 계산
4. **optimizer.step()**: 기울기로 가중치 갱신

```python
optimizer.zero_grad()              # 이전 기울기 초기화
prediction = model(x)              # 예측
loss = criterion(prediction, y)    # 손실 계산
loss.backward()                    # 기울기 계산
optimizer.step()                   # 가중치 갱신
```

> ⚠️ `loss.backward()`는 기울기만 계산한다. 가중치를 실제로 바꾸는 것은 `optimizer.step()`이다.

### 기본 용어

| 용어 | 뜻 | 설명 |
|---|---|---|
| Forward | 순전파 | 입력을 통해 예측값 계산 |
| Loss | 손실값 | 예측과 정답의 차이 |
| Backward | 역전파 | 가중치 조정에 필요한 기울기 계산 |
| Gradient | 기울기 | 가중치를 어느 방향으로 얼마나 바꿀지 알려주는 값 |
| Parameter | 매개변수 | 모델이 학습하면서 조정하는 값 (Weight, Bias 포함) |
| Weight | 가중치 | 입력 특성의 영향력을 조절하는 값 |
| Bias | 편향 | 결과에 더해지는 추가 학습 값 |
| Autograd | 자동 미분 | PyTorch가 기울기를 자동 계산하는 기능 |

---

## 2. Optimizer, Learning Rate, Scheduler

### Optimizer (최적화 알고리즘)

계산된 기울기로 **가중치를 어떤 규칙으로 바꿀지** 정하는 도구.

| 종류 | 특징 |
|---|---|
| **SGD** | Batch에서 계산한 기울기로 갱신. 구조가 단순함 |
| **SGD + Momentum** | 이전 이동 방향을 누적해서 사용 |
| **Adam** | 기울기의 평균과 제곱 평균으로 갱신 크기를 조절 |
| **AdamW** | Adam에서 Weight Decay를 분리해 적용. 초기 수렴이 빠른 편 |

```python
# SGD
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9
)

# AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

| 인자 | 의미 |
|---|---|
| `model.parameters()` | 학습할 매개변수 |
| `lr` | 학습률 |
| `momentum` | 이전 이동 방향의 누적 정도 |
| `weight_decay` | 가중치 감쇠 정도 |
| `betas` | Adam 계열의 이동평균 계수 |
| `eps` | 수치 계산 안정성을 위한 작은 값 |

> 어떤 Optimizer가 항상 더 좋은 것은 아니다. 같은 조건에서 실험하고 결과를 비교해야 한다.

### Learning Rate (학습률)

가중치를 한 번 갱신할 때 **얼마나 크게 이동할지** 정하는 값.

| 상태 | 결과 |
|---|---|
| 너무 작음 | 학습이 느림 |
| 적절함 | 안정적으로 Loss 감소 |
| 너무 큼 | 최적점을 지나치거나 Loss가 진동·발산 |

### Scheduler (학습률 조절기)

학습 진행 상황에 따라 **학습률을 바꿔 주는 도구**. 대표적으로 `ReduceLROnPlateau`가 있으며, 검증 성능이 정체(Plateau)되면 학습률을 줄인다.

| 용어 | 뜻 | 설명 |
|---|---|---|
| Plateau | 성능 개선 정체 | 검증 지표가 일정 기간 개선되지 않는 상태 |
| ReduceLROnPlateau | 정체 시 학습률 감소 | 성능 개선이 멈추면 학습률을 낮춤 |
| Factor | 감소 배율 | 학습률에 곱하는 값 (0.01 → 0.005) |
| Patience | 대기 횟수 | 개선 없이 기다리는 횟수 |
| Min LR | 최소 학습률 | 허용하는 최소 학습률 |

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4
)

# 검증 손실을 확인한 다음 호출
scheduler.step(valid_loss)
```

**한 줄 구분**

- Optimizer: 가중치를 **어떻게** 바꿀까?
- Scheduler: 학습률을 **언제, 얼마나** 바꿀까?

---

## 3. Batch · Iteration · Epoch

| 용어 | 뜻 | 설명 |
|---|---|---|
| Dataset | 데이터셋 | 전체 데이터 |
| Sample | 샘플 | 데이터 한 건 |
| Batch | 배치 | 한 번에 처리하는 데이터 묶음 |
| Batch Size | 배치 크기 | 한 배치에 포함된 데이터 수 |
| Iteration | 반복 횟수 | Batch 하나로 `optimizer.step()`까지 수행한 횟수 |
| Epoch | 에폭 | 학습 데이터 전체를 한 번 순회한 단위 |

### 예시: 데이터 240건, Batch Size 32

- Batch 1~7: 각 32건
- 마지막 Batch: 16건
- Iteration: 총 8회 (= 1 Epoch)

### DataLoader 인자

| 인자 | 의미 |
|---|---|
| `dataset` | 사용할 데이터셋 |
| `batch_size` | 배치 크기 |
| `shuffle` | 데이터 순서를 섞을지 여부 |
| `drop_last` | 마지막에 남은 작은 배치를 버릴지 여부 |
| `num_workers` | 데이터 로딩에 사용할 작업 프로세스 수 |

> `drop_last=False`라면 마지막 Batch의 크기가 다른 Batch보다 작을 수 있다.

---

## 4. Train · Validation · Test

학습에 사용한 데이터만으로 모델 성능을 판단하지 않기 위해 데이터를 세 가지로 나눈다.

| 구분 | 뜻 | 역할 |
|---|---|---|
| **Train** | 학습 데이터 | 모델의 가중치를 학습 |
| **Validation (Valid)** | 검증 데이터 | 모델 설정과 학습 중단 시점 선택 |
| **Test** | 테스트 데이터 | 최종 모델 성능 평가 |

예: 전체 20건 → Train 12건 / Valid 4건 / Test 4건

> 🔑 **중요 원칙**: Test 데이터는 모델을 선택하는 과정에서 사용하지 않고, 마지막 최종 평가를 위해 보관한다.

---

## 5. Overfitting vs Underfitting

- **Overfitting (과적합)**: 학습 데이터에 지나치게 맞춰져 새로운 데이터에서는 성능이 떨어지는 현상
- **Underfitting (과소적합)**: 충분히 학습하지 못해 학습 데이터조차 제대로 예측하지 못하는 현상

| 구분 | Train Loss | Valid Loss |
|---|---|---|
| 정상적인 학습 | 감소 | 감소 |
| Overfitting | 감소 | 최저점 이후 **증가** |
| Underfitting | 높거나 충분히 감소하지 않음 | 높거나 충분히 감소하지 않음 |

### Learning Curve (학습 곡선)

Epoch가 진행됨에 따라 Train과 Valid의 Loss가 어떻게 변하는지 나타낸 그래프.

- Train과 Valid가 함께 감소 → 학습이 개선되고 있는 상태
- Train은 감소, Valid는 증가 → **과적합 의심**
- 둘 다 높음 → **과소적합 의심**
- Loss가 크게 진동 → Learning Rate가 크거나 Batch가 작을 가능성
- NaN 발생 → 수치 불안정. 입력 또는 Loss 계산 등을 점검

---

## 6. Regularization (규제)

모델이 학습 데이터에 지나치게 의존하지 않도록 제한하는 기법.

| 기법 | 무엇을 조절하는가? | 주목적 |
|---|---|---|
| Dropout | 일부 뉴런의 출력 | 특정 특징에 대한 의존 감소 |
| BatchNorm | 중간 출력값의 분포 | 학습 안정화 |
| Weight Decay | 가중치 크기 | 큰 가중치 억제 |

### 6-1. Dropout

학습 중 일부 뉴런의 출력을 무작위로 0으로 만드는 기법.

- 특정 뉴런에 지나치게 의존하는 것을 방지한다.
- 과적합 완화에 도움을 준다.
- **학습 중에만 적용**되고, 평가 모드에서는 비활성화된다.

```python
nn.Dropout(p=0.5)   # p: 출력을 0으로 만들 확률
```

| 용어 | 의미 |
|---|---|
| `p` | 출력을 0으로 만들 확률 |
| Mask | 무작위로 적용하는 제거 패턴 |
| Activation | 뉴런의 활성화 출력값 |

### 6-2. BatchNorm (Batch Normalization)

배치 내부의 통계(평균·분산)를 이용해 중간 출력값을 정규화하는 기법. 중간값의 분포를 안정화해 학습을 원활하게 한다.

```python
nn.Linear(2, 64),
nn.BatchNorm1d(64),
nn.ReLU()
```

| 용어 | 뜻 | 설명 |
|---|---|---|
| BatchNorm | 배치 정규화 | 배치 통계로 중간 출력값 정규화 |
| Mean | 평균 | 데이터의 평균값 |
| Variance | 분산 | 데이터가 평균에서 얼마나 퍼져 있는지 |
| Running Mean | 누적 평균 | 학습 중 누적한 평균 |
| Running Variance | 누적 분산 | 학습 중 누적한 분산 |
| `num_features` | 특성 수 | 입력의 Feature 개수 |
| `eps` | 엡실론 | 수치 계산 안정성을 위한 작은 값 |

**train / eval 차이**

| 구분 | `model.train()` | `model.eval()` |
|---|---|---|
| 사용하는 통계 | 현재 Batch 통계 | 누적 통계 |
| 통계 갱신 | O | X |

- 일반적인 배치 위치: `Linear → BatchNorm1d → ReLU`
- `num_features`는 앞 `Linear`의 `out_features`와 같아야 한다.
- Batch 통계를 사용하므로 학습 Batch Size가 1이면 문제가 생길 수 있다.
- Dropout, Weight Decay와 목적이 다르다. 핵심은 배치 통계를 이용한 **중간 출력값 정규화와 학습 안정화**다.

### 6-3. Weight Decay (가중치 감쇠)

모델의 가중치가 지나치게 커지는 것을 억제하는 기법.

- 큰 가중치에 불이익을 준다.
- 모델이 특정 입력에 과도하게 반응하는 것을 억제한다.
- 과적합 완화에 도움을 줄 수 있다.
- 너무 크게 설정하면 **Underfitting**이 발생할 수 있다.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

---

## 7. train() · eval() · no_grad()

PyTorch에서 모델의 동작 모드와 기울기 계산 여부를 제어하는 기능.

| 명령어 | 역할 |
|---|---|
| `model.train()` | 학습 모드 |
| `model.eval()` | 평가 모드 |
| `torch.no_grad()` | 기울기 계산 비활성화 |
| `torch.inference_mode()` | 추론에 최적화된 기울기 추적 비활성화 |

```python
# 학습 모드
model.train()

# 평가 모드
model.eval()

# 평가 중 기울기 계산 방지
with torch.no_grad():
    prediction = model(x)
```

**주의할 점**

- `train()`과 `eval()`은 모델의 **동작 방식**(Dropout, BatchNorm 등)을 바꾼다.
- `eval()`만 호출한다고 기울기 계산이 자동으로 꺼지는 것은 아니다.
- 기울기 계산을 끄려면 `no_grad()` 또는 `inference_mode()`를 별도로 사용한다.

---

## 8. Early Stopping (조기 종료)

검증 데이터의 성능 개선이 멈추면 학습을 조기에 종료하는 기법.

| 용어 | 뜻 | 설명 |
|---|---|---|
| Early Stopping | 조기 종료 | 개선이 멈추면 학습 종료 |
| Best | 최고 성능 | 지금까지 가장 좋았던 검증 결과 |
| Patience | 인내 횟수 | 개선 없이 허용할 Epoch 수 |
| Min Delta | 최소 개선량 | 개선으로 인정할 최소 변화 |
| Best State | 최고 상태 | 가장 성능이 좋았던 모델 상태 |
| State Dict | 상태 사전 | 모델의 Parameter 등을 저장하는 객체 |

### 동작 예시 (Patience = 2)

| Epoch | Valid Loss | 판단 |
|---:|---:|---|
| 1 | 0.65 | 최고 기록 저장 |
| 2 | 0.50 | 최고 기록 갱신 |
| 3 | 0.42 | 최고 기록 갱신 |
| 4 | 0.45 | 대기 1회 |
| 5 | 0.47 | 대기 2회 |
| 6 | 0.46 | 학습 중단 |

Epoch 6에서 학습을 멈췄지만 가장 좋은 모델은 **Epoch 3**의 모델이다. 따라서 학습 종료 후에는 Epoch 3의 모델 상태를 **복원**해야 한다.

> **중요**: 학습을 중단하는 판단과 최고 성능 모델을 저장·복원하는 작업은 별도로 구현해야 한다.

---

## 9. 실험·평가 용어

| 용어 | 뜻 | 설명 |
|---|---|---|
| Accuracy | 정확도 | 전체 예측 중 정답 비율 |
| Loss | 손실값 | 예측과 정답의 차이를 측정 |
| Valid Loss | 검증 손실 | 검증 데이터의 손실 |
| Valid Acc | 검증 정확도 | 검증 데이터의 정확도 |
| Seed | 시드 | 난수 생성의 기준값 |
| Split | 분할 | 데이터를 학습·검증·테스트로 나누는 것 |
| Baseline | 기준선 | 다른 실험과 비교하기 위한 기본 설정 |
| History | 학습 기록 | Epoch별 Loss, Accuracy 등을 저장 |
| Checkpoint | 체크포인트 | 학습 중 특정 시점의 모델 상태 저장 |
| Reproducibility | 재현성 | 같은 조건에서 실험 결과를 재현하는 정도 |
| NaN | 숫자가 아님 | 유효한 수치가 아닌 값 |
| Inf | 무한대 | 무한대 값이 발생한 상태 |

### 공정한 실험 비교 원칙

Optimizer나 Learning Rate를 비교할 때는 비교 대상 외의 조건을 동일하게 유지한다.

- 동일한 데이터 분할
- 동일한 모델 구조
- 동일한 Seed
- 동일한 Epoch

조건이 달라지면 Optimizer의 효과와 다른 요인의 효과를 구분하기 어렵다.

---

## 10. 전체 학습 흐름

1. **데이터 분할**: Train / Validation / Test
2. **학습 조건 설정**: Optimizer / Learning Rate / Batch Size
3. **모델 학습**: Forward → Loss → Backward → Step
4. **학습 상태 확인**: Train Loss / Valid Loss / Accuracy
5. **문제 진단 및 개선**: Dropout / BatchNorm / Weight Decay / Scheduler
6. **학습 종료 및 모델 선택**: Early Stopping / Best State 복원
7. **최종 평가**: Test 데이터로 성능 확인

---

## 11. 복습용 요약

### 개념별 핵심 질문

| 개념 | 핵심 질문 | 역할 |
|---|---|---|
| Optimizer | 어떻게 바꿀까? | 가중치 갱신 |
| Learning Rate | 얼마나 바꿀까? | 갱신 크기 결정 |
| Scheduler | 언제 학습률을 바꿀까? | 학습률 조절 |
| Batch Size | 한 번에 몇 건? | 데이터 처리 단위 |
| Iteration | 몇 번 갱신했나? | Batch 학습 횟수 |
| Epoch | 전체 데이터를 몇 번 봤나? | 전체 순회 횟수 |
| Dropout | 특정 뉴런에 의존하지 않게 하려면? | 일부 출력 제거 |
| BatchNorm | 중간값을 안정화하려면? | 배치 통계 정규화 |
| Weight Decay | 가중치가 너무 커지지 않게 하려면? | 가중치 감쇠 |
| Early Stopping | 언제 학습을 멈출까? | 조기 종료 |

### 우선 이해할 개념

1. Optimizer와 Learning Rate의 차이
2. Batch / Iteration / Epoch의 차이
3. Train / Valid / Test의 차이
4. Overfitting과 Underfitting의 차이
5. Dropout과 BatchNorm의 차이
6. `train()`과 `eval()`의 차이
7. Early Stopping의 작동 원리

### 추가로 익힐 개념

- SGD와 AdamW의 내부 차이
- Scheduler의 세부 인자
- BatchNorm의 Running Mean과 Variance
- Weight Decay의 수학적 원리

### Day 10에서 가장 중요한 세 가지

- **Optimizer와 Learning Rate**: 모델을 어떻게 학습시킬 것인가?
- **Overfitting과 Regularization**: 모델이 왜 새로운 데이터에서 성능이 떨어지는가?
- **Validation과 Early Stopping**: 어떤 모델을 최종적으로 선택할 것인가?
