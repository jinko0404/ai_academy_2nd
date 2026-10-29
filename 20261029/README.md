# 🧠 Day 9 딥러닝 핵심 개념 정리

> **GPU · Tensor · ANN · 모델 학습**
>
> 기준 자료: `Day9_GPU_Tensor_ANN_20261029_v2.pdf`
> 수업 PPT의 핵심 용어와 코드 흐름을 복습하기 위한 README입니다.

## 📚 목차

1. [GPU와 실행 환경](#1-gpu와-실행-환경)
2. [Tensor와 데이터 모양](#2-tensor와-데이터-모양)
3. [자동 미분과 기울기](#3-자동-미분과-기울기)
4. [ANN과 신경망 구조](#4-ann과-신경망-구조)
5. [Logits · Softmax · 손실함수](#5-logits--softmax--손실함수)
6. [Dataset · DataLoader · Batch](#6-dataset--dataloader--batch)
7. [모델 학습 루프](#7-모델-학습-루프)
8. [평가와 결과 해석](#8-평가와-결과-해석)
9. [우선순위별 복습 목록](#9-우선순위별-복습-목록)

> 💡 **한 문장으로 기억하기**
>
> **Tensor**로 데이터를 준비하고 → **ANN**으로 예측하고 → **Loss**로 오차를 계산하고 → **Backward**로 기울기를 구하고 → **Optimizer**로 파라미터를 갱신한다.

---

## 1. GPU와 실행 환경

| 용어 | 뜻 |
|---|---|
| CPU | 컴퓨터의 중앙처리장치. 다양한 종류의 작업과 프로그램 흐름을 처리한다. |
| GPU | 많은 수치 연산을 병렬로 처리할 수 있어 딥러닝 계산에 유리한 장치다. |
| VRAM | GPU가 사용하는 전용 메모리. 모델 파라미터, 입력 데이터, 중간 계산 결과 등을 저장한다. |
| CUDA | NVIDIA GPU에서 계산을 실행하기 위한 기술 및 소프트웨어 환경이다. |
| GPU Driver | 운영체제와 GPU가 통신하도록 지원하는 소프트웨어다. |
| PyTorch | Tensor 연산, 자동 미분, 신경망 구성과 학습을 지원하는 딥러닝 라이브러리다. |
| `device` | 연산을 수행할 장치. 주로 `cpu` 또는 `cuda`를 사용한다. |

### 환경 확인

```python
import torch

print(torch.__version__)             # 설치된 PyTorch 버전
print(torch.cuda.is_available())     # PyTorch에서 CUDA 사용이 가능한지 확인

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # 첫 번째 GPU 이름
```

- `nvidia-smi`: GPU, 드라이버, 메모리 사용량 등을 확인하는 명령이다.

> ⚠️ **주의:** GPU가 컴퓨터에 있다는 것과 PyTorch에서 CUDA를 사용할 수 있다는 것은 **서로 다른 확인 항목**이다.

### 장치 이동

모델과 입력 데이터는 **같은 장치**에 있어야 연산할 수 있다.

```python
model = model.to(device)  # 모델을 지정한 장치로 이동
X = X.to(device)          # 입력 Tensor를 같은 장치로 이동
```

---

## 2. Tensor와 데이터 모양

**Tensor(텐서)** 는 딥러닝에서 숫자 데이터를 저장하고 계산하는 기본 자료구조다.

### 속성

| 용어 | 뜻 | 확인 예시 |
|---|---|---|
| `shape` | 데이터의 모양과 각 차원의 크기 | `X.shape` |
| `dtype` | 데이터의 자료형 | `X.dtype` |
| `device` | 데이터가 위치한 장치 | `X.device` |
| `requires_grad` | 연산의 기울기를 추적할지 여부 | `X.requires_grad` |

### 생성

| 함수 | 뜻 | 사용 예시 |
|---|---|---|
| `torch.tensor()` | 주어진 값으로 Tensor를 생성한다. | `torch.tensor([1., 2.])` |
| `torch.randn()` | 정규분포에서 난수를 뽑아 Tensor를 만든다. | `torch.randn(2, 3)` |
| `torch.zeros()` | 0으로 채운 Tensor를 만든다. | `torch.zeros(2, 3)` |
| `torch.ones()` | 1로 채운 Tensor를 만든다. | `torch.ones(2, 3)` |

### 모양 변경

| 함수 | 뜻 | 사용 예시 |
|---|---|---|
| `reshape()` | 원소 수를 유지하면서 모양을 바꾼다. | `values.reshape(3, 4)` |
| `transpose()` | 차원의 순서를 바꾼다. | `X.transpose(0, 1)` |
| `unsqueeze()` | 크기가 1인 차원을 추가한다. | `X.unsqueeze(0)` |
| `squeeze()` | 크기가 1인 차원을 제거한다. | `X.squeeze()` |

### 값 꺼내기 · 이동

| 함수 | 뜻 | 사용 예시 |
|---|---|---|
| `.item()` | 원소 하나를 Python 숫자로 가져온다. | `loss.item()` |
| `.tolist()` | Tensor 값을 Python 리스트로 변환한다. | `X.tolist()` |
| `.to(device)` | Tensor를 지정한 장치로 이동한다. | `X.to(device)` |

### Shape 읽기

```python
X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [0.0, 1.0]
])

print(X.shape)  # torch.Size([3, 2])
```

`(3, 2)` = **샘플 3개, 샘플마다 특성 2개**

| 용어 | 뜻 |
|---|---|
| Sample (샘플) | 데이터 한 건 |
| Feature (특성) | 샘플 하나를 설명하는 입력 정보 |
| `torch.float32` | 수치 입력 데이터에 자주 사용하는 자료형 |
| `torch.long` | 분류 정답을 클래스 번호로 표현할 때 일반적으로 사용하는 자료형 |

---

## 3. 자동 미분과 기울기

| 용어 | 뜻 |
|---|---|
| Autograd | PyTorch가 연산을 추적하고 기울기를 자동으로 계산하는 기능이다. |
| 연산 그래프 | 어떤 연산을 거쳐 결과가 나왔는지 나타내는 계산 관계다. |
| Gradient (기울기) | 입력이나 파라미터가 변할 때 손실이 어떻게 변하는지 나타내는 값이다. |
| `requires_grad=True` | 관련 연산의 기울기를 추적하도록 설정한다. |
| `backward()` | 역전파를 수행해 기울기를 계산한다. |
| `.grad` | 계산된 기울기가 저장되는 속성이다. |
| Gradient Descent | 기울기를 이용해 손실이 작아지는 방향으로 파라미터를 갱신하는 방법이다. |
| Learning Rate (`lr`) | 한 번 갱신할 때 이동하는 크기를 조절하는 값이다. |
| `torch.no_grad()` | 블록 안의 연산을 자동 미분 그래프에 기록하지 않는다. |
| `detach()` | 기존 자동 미분 그래프에서 분리된 Tensor를 반환한다. |
| `zero_()` | Tensor의 값을 0으로 바꾼다. `w.grad.zero_()`는 저장된 기울기를 초기화한다. |

### 수동 Gradient Descent

```python
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

loss = (w - 4.0) ** 2  # 현재 w가 목표값 4에서 얼마나 벗어났는지 계산
loss.backward()        # 기울기 계산
print(w.grad)          # 현재 기울기 확인

with torch.no_grad():
    w -= lr * w.grad   # 기울기를 이용해 기존 w를 갱신

w.grad.zero_()         # 다음 반복에 이전 기울기가 누적되지 않도록 초기화
```

**학습 흐름**

1. 손실(`loss`)을 계산한다.
2. `loss.backward()`로 기울기를 계산한다.
3. `w -= lr * w.grad`로 값을 갱신한다.
4. `w.grad.zero_()`로 기울기를 초기화한다.

> ⚠️ **중요:** `backward()`는 기울기를 **계산**하는 단계이며, 그 자체로 파라미터를 갱신하지 않는다. 파라미터 갱신은 직접 수행하거나 Optimizer를 사용한다.

---

## 4. ANN과 신경망 구조

**ANN (Artificial Neural Network, 인공신경망)** 은 입력 데이터를 여러 층에서 변환해 결과를 만드는 모델이다.

### 구성 요소

| 용어 | 뜻 |
|---|---|
| Neuron (뉴런) | 입력을 받아 가중합 등을 계산하는 인공 신경 단위다. |
| Layer (층) | 뉴런들이 모여 계산을 수행하는 단계다. |
| 입력층 | 데이터를 모델에 전달하는 부분이다. |
| 은닉층 (Hidden Layer) | 입력과 출력 사이에서 중간 계산을 수행하는 부분이다. |
| 출력층 | 최종 예측 결과를 만드는 부분이다. |
| Weight (가중치) | 입력 정보가 결과에 미치는 영향의 크기를 학습하는 값이다. |
| Bias (편향) | 가중합에 더해지는 보정값이다. |
| Activation Function (활성화 함수) | 층 사이에 비선형성을 추가하는 함수다. |
| ReLU | 음수는 0으로, 양수는 그대로 통과시키는 활성화 함수다. |

### PyTorch 클래스

| 이름 | 뜻 |
|---|---|
| `nn.Linear(a, b)` | 입력 특성 `a`개를 받아 출력 특성 `b`개로 변환한다. |
| `nn.Sequential` | 여러 신경망 층을 작성한 순서대로 연결한다. |
| `nn.Module` | 사용자 정의 신경망을 만들 때 상속하는 PyTorch 기본 클래스다. |
| `forward()` | 입력이 모델을 통과해 출력으로 계산되는 과정을 정의한다. |

### 2 → 8 → 2 신경망

```python
model = nn.Sequential(
    nn.Linear(2, 8),  # 입력 특성 2개 → 출력 특성 8개
    nn.ReLU(),        # 활성화 함수 적용
    nn.Linear(8, 2)   # 입력 특성 8개 → 최종 출력 2개
)
```

입력 데이터가 `(N, 2)` 형태라면 출력도 `(N, 2)` 형태다. (`N` = 데이터 개수)

### 학습 가능한 파라미터 수: 42개

| 층 | 가중치 | 편향 | 합계 |
|---|---|---|---|
| `nn.Linear(2, 8)` | 16 | 8 | 24 |
| `nn.ReLU()` | 없음 | 없음 | 0 |
| `nn.Linear(8, 2)` | 16 | 2 | 18 |
| **총합** | | | **42** |

### 사용자 정의 Module

```python
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)
```

| 코드 | 역할 |
|---|---|
| `super().__init__()` | 부모 클래스 `nn.Module`의 초기화 작업을 수행한다. |
| `self.net` | 신경망 층들을 묶어 둔 모델이다. |
| `forward(self, x)` | 입력 `x`를 신경망에 전달하고 결과를 반환한다. |

- `nn.Sequential`로 만든 모델과 사용자 정의 `nn.Module` 모델은 **구조와 파라미터가 같다면 같은 출력**을 낼 수 있다.

---

## 5. Logits · Softmax · 손실함수

| 용어 | 뜻 |
|---|---|
| Logits | 모델이 계산한 클래스별 원시 점수. 합이 1일 필요가 없다. |
| Softmax | logits를 합이 1인 확률 분포로 변환한다. |
| Predicted Class | 가장 높은 점수를 받은 클래스 번호다. |
| Loss (손실) | 모델의 예측과 정답 사이의 차이를 측정하는 값이다. |
| `CrossEntropyLoss` | 분류 문제에서 사용하는 대표적인 손실함수다. |
| Accuracy (정확도) | 전체 예측 중 정답을 맞힌 비율이다. |

### Logits와 Softmax의 차이

| 단계 | 예시 값 |
|---|---|
| logits | `[1.8, -0.4]` |
| Softmax 적용 후 | `[0.90, 0.10]` |
| 예측 클래스 (가장 큰 값의 인덱스) | `0` |

### 분류 학습 코드

```python
logits = model(X)                  # 모델의 원시 점수
loss = loss_fn(logits, y)          # logits와 정답을 비교해 손실 계산
predicted = logits.argmax(dim=1)   # 각 데이터에서 가장 높은 점수의 클래스 번호
```

> ⚠️ **중요:** `nn.CrossEntropyLoss()`를 사용하는 일반적인 분류 학습에서는 Softmax를 미리 적용하지 않고 **logits를 직접 전달**한다. Softmax는 확률로 해석할 때 별도로 사용할 수 있다.

---

## 6. Dataset · DataLoader · Batch

| 용어 | 뜻 |
|---|---|
| Dataset | 학습에 사용할 데이터와 정답을 구성한 자료다. |
| `TensorDataset` | 입력 Tensor와 정답 Tensor를 같은 인덱스로 묶는다. |
| `DataLoader` | Dataset을 반복해서 읽고 배치로 나누어 공급한다. |
| Sample (샘플) | 데이터 한 건이다. |
| Feature (특성) | 샘플 하나를 설명하는 입력 정보다. |
| Target / Label | 모델이 예측해야 하는 정답이다. |
| Batch | 한 번의 학습 단계에서 처리하는 데이터 묶음이다. |
| `batch_size` | 배치 하나에 들어가는 샘플 수다. |
| `shuffle=True` | 데이터를 공급할 때 순서를 섞는다. |

### 사용 예시

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

xb, yb = next(iter(loader))
```

| 변수 | 의미 |
|---|---|
| `X` / `y` | 입력 데이터 / 정답 데이터 |
| `xb` / `yb` | 현재 배치의 입력 / 정답 |

- 샘플이 200개이고 `batch_size=32`라면, 마지막 배치는 남은 샘플 수에 따라 32개보다 작을 수 있다. (200 = 32 × 6 + 8)

---

## 7. 모델 학습 루프

학습 루프는 데이터를 모델에 넣고 → 손실을 계산하고 → 기울기를 구하고 → 파라미터를 갱신하는 과정을 **반복**한다.

### 한 step의 순서

| 순서 | 코드 | 역할 |
|---|---|---|
| 1 | `optimizer.zero_grad()` | 이전 기울기 초기화 |
| 2 | `logits = model(xb)` | 순전파 (Forward Pass), 예측 점수 계산 |
| 3 | `loss = loss_fn(logits, yb)` | 손실 계산 |
| 4 | `loss.backward()` | 역전파 (Backward Pass), 기울기 계산 |
| 5 | `optimizer.step()` | 파라미터 갱신 |

### 전체 흐름

```python
for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)

    optimizer.zero_grad()
    logits = model(xb)
    loss = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()
```

### 학습 관련 용어

| 용어 | 뜻 |
|---|---|
| Optimizer | 기울기를 이용해 모델 파라미터를 갱신하는 도구다. |
| SGD | 확률적 경사 하강법. 대표적인 Optimizer다. |
| `zero_grad()` | 기존 기울기를 초기화한다. |
| `step()` | 파라미터 갱신을 한 번 수행한다. |
| `model.train()` | 모델을 학습 모드로 전환한다. |
| `model.eval()` | 모델을 평가 모드로 전환한다. |
| `torch.no_grad()` | 불필요한 기울기 계산 그래프 생성을 막는다. |
| Epoch | 학습 데이터 전체를 한 번 사용하는 단위다. |
| Training History | 학습 중 손실, 정확도 등의 기록이다. |
| Checkpoint | 모델을 저장해 나중에 복원하거나 학습을 이어갈 수 있게 하는 파일이다. |

> ⚠️ **주의:** `model.eval()`과 `torch.no_grad()`는 **서로 다른 기능**이다. 평가할 때는 보통 둘을 함께 사용한다.

---

## 8. 평가와 결과 해석

| 용어 | 뜻 |
|---|---|
| Train Set | 모델 파라미터를 학습하는 데이터다. |
| Validation Set (Valid) | 모델이나 학습 설정을 비교·선택하는 데이터다. |
| Test Set | 최종 성능을 평가하는 데이터다. |
| Generalization (일반화) | 학습에 사용하지 않은 새로운 데이터에서도 잘 예측하는 능력이다. |
| Overfitting (과적합) | 학습 데이터에는 잘 맞지만 새로운 데이터에서는 성능이 떨어지는 현상이다. |
| Loss | 예측과 정답 사이의 차이를 측정한다. |
| Accuracy | 예측 중 정답을 맞힌 비율이다. |
| Reproducibility (재현성) | 같은 조건에서 실험을 다시 수행하고 결과를 비교할 수 있는 성질이다. |

**결과를 해석할 때 기억할 점**

- Loss가 줄어드는 것만으로 좋은 모델이라고 단정할 수 없다.
- Accuracy와 Loss는 서로 다른 지표다.
- 표본 수와 평가 데이터가 무엇인지 함께 확인한다.
- Test Set은 최종 평가에만 사용하고, 반복적인 모델 선택에는 사용하지 않는 것이 좋다.

---

## 9. 우선순위별 복습 목록

### 🥇 1순위 — 지금 코드 읽기에 필수

- [ ] `Tensor`, `shape`, `dtype`, `device`
- [ ] `nn.Linear`, `nn.ReLU`, `nn.Sequential`
- [ ] `nn.Module`, `forward()`
- [ ] `logits`, `loss`, `gradient`
- [ ] `backward()`, `torch.no_grad()`

### 🥈 2순위 — 학습 루프 이해

- [ ] `zero_grad()`
- [ ] `Optimizer`, `SGD`, `step()`
- [ ] `CrossEntropyLoss`
- [ ] `Dataset`, `DataLoader`, `Batch`, `Epoch`

### 🥉 3순위 — 환경과 결과 해석

- [ ] `GPU`, `CUDA`, `VRAM`
- [ ] `model.train()`, `model.eval()`
- [ ] `Train / Valid / Test`
- [ ] `Loss`, `Accuracy`, `Overfitting`
- [ ] 결과 저장과 재현성
