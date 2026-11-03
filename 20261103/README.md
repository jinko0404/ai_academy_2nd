# Day 12. 딥러닝 핵심 아키텍처 및 학습 (Sequence 모델)

> **학습일:** 2026-11-03 (자료 제목·커리큘럼 기준)
>
> **주제:** Sequence Tensor → RNN·LSTM → Autoencoder → Transformer → 미니 모델
>
> **기준 자료:** `20261102_2.pdf`

---

## 1. 오늘 수업의 전체 흐름

지난 수업(Day 11)은 CNN으로 **공간 구조**를 다뤘고, 오늘은 **시간·순서·재구성·Attention 구조**로 확장한다.

| 모델 | 역할 |
|------|------|
| RNN | 이전 Hidden state를 다음 시점으로 전달해 순서를 처리한다. |
| LSTM | Gate와 Cell state로 긴 의존성을 더 안정적으로 보존한다. |
| Autoencoder | 입력을 작은 Latent 표현으로 압축한 뒤 다시 복원한다. |
| Transformer | Attention으로 입력 위치들을 직접 비교해 문맥을 만든다. |

### 1-1. 센서 기록을 사용하는 두 가지 업무 목적

가상 업무 예시: 설비 센서 관측 기록으로 **다음 값을 예상**하거나, 현재 패턴의 **점검 후보를 찾는다**.

| 목적 | 모델 | 입력 | 학습 정답 | 활용 |
|------|------|------|-----------|------|
| ① 다음 값 예측 | LSTM | 과거 12개 관측 | 다음 시점의 실제 값 | 운영 판단에 참고 |
| ② 입력 재구성 | Autoencoder | 현재의 8개 특성 | 원래 입력 | 오차가 큰 사례 점검 |

수업에서는 사인파 예측과 8차원 데이터 재구성으로 같은 흐름을 연습한다.

### 1-2. 전체 지도에서 오늘 모델의 위치

- 순서 자료의 예측·분류 → **RNN / LSTM**, **Transformer Encoder**
- 입력 재구성 → 오늘의 **MLP형 Autoencoder** (Encoder · Latent · Decoder)
- Autoencoder는 재구성 학습 구성이며, 오늘은 Linear 층으로 만든다.

### 1-3. 8교시 운영

| 교시 | 핵심 내용 | 설명·시연 | 실습·리뷰 |
|-----:|-----------|----------:|----------:|
| 1 | Sequence Tensor·Padding·Task 유형 | 30분 | 20분 |
| 2 | RNN·Hidden state·순차 예측 | 25분 | 25분 |
| 3 | LSTM·Gate·시계열 예측 | 25분 | 25분 |
| 4 | Autoencoder·재구성·이상 탐지 | 25분 | 25분 |
| 5 | Attention·Transformer Encoder | 25분 | 25분 |
| 6 | 아키텍처 비교·Mask·미니 모델 | 25분 | 25분 |
| 7 | 길이·Padding·CPU/GPU 성능 비교 | 25분 | 25분 |
| 8 | 아키텍처 비교 프로젝트·시연 | 20분 | 30분 |

### 1-4. 오늘 범위와 다음 수업의 경계

| 오늘 · 11월 3일 | 다음 수업 · 11월 4일 |
|-----------------|----------------------|
| RNN·LSTM으로 순서와 장기 의존성 처리 | OpenCV 이미지 표현·색공간·리사이즈 |
| Autoencoder의 압축·재구성과 오류 해석 | 필터·형태학 처리와 영상 입출력 |
| Transformer Encoder의 Attention·Mask | 프레임 처리·데이터 증강 파이프라인 |
| 아키텍처 비교도와 시계열·재구성 미니 모델 | 원본 대비 전처리 결과 검증 |

### 1-5. 실습 환경 확인

- torch import와 CPU/GPU 장치를 확인하고 PyTorch 실제 설치 버전을 기록한다.
- Sequence Tensor의 기본 Shape는 `[Batch, Time, Feature]`로 통일한다.
- 서로 다른 길이는 Padding·lengths·Mask로 실제 값과 채움 위치를 구분한다.
- 정상 설치 환경을 유지한다. 기본 실습은 CPU로, GPU 비교는 사용 가능할 때 진행한다.

---

## 2. Sequence 데이터와 Tensor

### 2-1. 독립 표본 vs 순서가 있는 데이터

| 독립 표본 | 순서가 있는 데이터 |
|-----------|--------------------|
| 한 행 또는 한 이미지가 독립적인 샘플이다. | 시점·토큰의 순서가 의미에 직접 관여한다. |
| 표본을 모아 Batch로 처리한다. | 고정 길이와 가변 길이 모두 가능하다. |
| 표본 간 순서와 표본 내부 순서를 구분한다. | 현재 출력이 이전 상태나 다른 위치를 참조한다. |
| 예: 표 데이터·고정 크기 이미지 | 예: 센서·문장·로그·오디오 |

### 2-2. Sequence Tensor의 세 축

| 축 | 의미 | 예 `[32, 50, 8]` |
|----|------|------------------|
| B | Batch (Sequence 개수) | 32 |
| T | Time (시점·토큰 수) | 50 |
| F | Feature (한 시점의 특성 수) | 8 |

- 기본 Shape는 `[B, T, F]`(BTF)로 통일한다. `batch_first=True`일 때의 형태다.
- 한 입력 묶음의 구조: Batch(Window B개) → Window(과거 시점 T개) → 한 시점(동시에 관측한 특성 F개).
- 이번 실습의 LSTM 입력은 `[B,12,1]`이다. B는 Window 개수, T=12는 과거 12개 관측, F=1은 시점마다 값 1개, 정답은 다음 값 `[B,1]`이다.

### 2-3. 가변 길이: Padding · lengths · Mask

- 짧은 Sequence에 **Padding**을 추가해 같은 T로 Batch를 만든다. Padding을 쓰면 T는 Batch 안의 최대 길이다.
- **lengths**는 샘플별 실제 길이다. Shape는 `[B]`.
- **Mask**는 실제 값과 Padding 위치를 구분해 Attention·Pooling에서 Padding을 제외한다.
- Padding 비율이 높으면 의미 없는 위치에도 연산·메모리를 쓰게 된다.
- Bucket·PackedSequence·Nested Tensor는 상황에 따라 효율을 높인다.

```python
import torch
x = torch.randn(4, 6, 3)
lengths = torch.tensor([6, 4, 5, 3])
mask = torch.arange(6)[None, :] >= lengths[:, None]   # True = Padding 위치
x = x.masked_fill(mask.unsqueeze(-1), 0.0)
print(x.shape)                  # torch.Size([4, 6, 3]) → 4개 Sequence·최대 길이 6·특성 3개
print(x.dtype, x.device)        # float32 → 모델 가중치와 같은 실수형
print(lengths.shape, lengths.dtype)   # lengths shape=[4] → 샘플별 실제 길이
```

### 2-4. 주요 함수: Padding과 PackedSequence

```python
pad_sequence(sequences, batch_first=False, padding_value=0.0)
pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)
```

| 인자 | 설명 |
|------|------|
| `sequences` / `input` | 가변 길이 Tensor 목록 또는 Padding된 Tensor (필수) |
| `lengths` | 각 샘플의 실제 길이. CPU Tensor 또는 정수 목록 (필수) |
| `batch_first` | True면 `[B,T,F]`, 기본은 False |
| `padding_value` | 채움 값 |
| `enforce_sorted` | 길이 정렬 여부 |
| `total_length` | `pad_packed_sequence`에서 복원할 전체 길이 |

- 반환: Padding Tensor 또는 PackedSequence. `batch_first=True`이면 Padding은 `[B,T,F]`, `Packed.data`는 `[Σ길이,F]`.
- `pad_packed_sequence`는 Padding Tensor와 lengths를 반환한다.
- RNN/LSTM은 PackedSequence를 입력·출력으로 사용할 수 있다.

### 2-5. Sequence 입출력의 대표 구조 (Task 유형)

| 유형 | 설명 |
|------|------|
| Many-to-one | 전체 Sequence를 읽고 최종 클래스·값 하나를 예측 |
| Many-to-many | 각 시점마다 태그·상태·예측을 출력 |
| Next-step | 과거 Window로 다음 시점의 값을 예측 |
| Encoder–Decoder | 입력 Sequence를 다른 길이의 출력 Sequence로 변환 |

---

## 3. RNN

### 3-1. 개념

- 같은 Cell을 시간축에 반복 적용해 Sequence를 읽는다.
- 각 시점의 입력 $x_t$와 이전 Hidden state $h_{t-1}$을 함께 사용한다.
- 같은 가중치를 모든 시점에 공유한다. 그래서 길이가 달라도 같은 Cell을 반복 적용할 수 있다.
- $h_t$는 현재까지 읽은 정보를 고정된 `hidden_size`로 요약한 것이다.

한 시점의 계산:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

### 3-2. output과 h_n 구분

| | output | h_n |
|--|--------|-----|
| 의미 | 모든 시점의 **마지막 층** 출력 | 모든 층·방향의 **마지막 상태** |
| Shape | `batch_first=True`: `[B,T,D×H]` | `[L×D,B,H]` |
| 용도 | 시점별 태깅·Attention 입력 | Sequence 분류의 최종 표현 |
| 주의 | 유효 시점 선택·Packing·Mask를 구분 | bidirectional이면 방향 D=2 |

- L: 층 수, D: 방향 수(양방향이면 2), H: `hidden_size`.

### 3-3. 주요 함수

```python
nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity="tanh",
       bias=True, batch_first=False, dropout=0.0, bidirectional=False)
output, h_n = rnn(input, hx=None)
```

| 인자 | 설명 |
|------|------|
| `input_size` | 시점별 입력 Feature 수 (필수) |
| `hidden_size` | Hidden state Feature 수 (필수) |
| `num_layers`·`batch_first`·`bidirectional` | 층 수·축 순서·양방향 |
| `nonlinearity`·`dropout`·`bias` | 활성화·층 사이 Dropout·Bias |
| `hx` | 초기 Hidden state. 생략하면 0. Shape `[L×D,B,H]` |

- 입력 `[B,T,F]` → output `[B,T,D×H]`, h_n `[L×D,B,H]`.
- `batch_first`는 input/output에만 적용되고 **h_n 축 순서는 고정**이다 (`hx`도 `batch_first`와 무관).

### 3-4. RNN Sequence 분류 모델의 Shape 흐름

```python
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(3, 8, batch_first=True)
        self.fc = nn.Linear(8, 2)
    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])
```

`[B,T,3]` → output `[B,T,8]` → `h_n[-1]` `[B,8]` → logits `[B,2]`

### 3-5. 기본 RNN의 장기 의존성 한계

- 역전파가 긴 시간축을 지나며 Gradient가 작아지거나 커질 수 있다.
- 초기 시점의 정보가 많은 갱신을 거치며 희미해질 수 있다.
- Sequence가 길수록 순차 계산 때문에 병렬화가 제한된다.
- 대안: LSTM·GRU의 Gate 또는 Transformer Attention.

---

## 4. LSTM

### 4-1. 개념

Gate와 Cell state로 장기 정보를 제어한다. 기억을 **지울지·쓸지·출력할지**를 학습한다.

- 기본 RNN은 h를 갱신하고, LSTM은 **h와 c를 함께** 전달한다.
- LSTM이 추가로 유지하는 상태는 **Cell state c**이다.

### 4-2. Gate의 역할

| 구성 | 역할 |
|------|------|
| Forget gate $f_t$ | 이전 기억 $c_{t-1}$을 얼마나 유지할지 0~1로 조절 |
| Input gate $i_t$ · Candidate $g_t$ | 새 정보를 기억에 얼마나 쓸지 조절 |
| Cell state $c_t$ | 장기 정보를 비교적 직접적인 경로로 전달 |
| Output gate $o_t$ | 현재 Hidden state로 노출할 정보를 결정 |

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)
$$

Cell state 한 원소의 계산 예시 (원리 설명용 숫자):

| 이전 기억에서 남길 양 | 새 정보에서 쓸 양 | 갱신한 기억 c |
|----------------------|-------------------|---------------|
| Forget gate × 이전 c = 0.5 × 2 = 1 | Input gate × 새 후보 = 0.4 × 0.5 = 0.2 | 1 + 0.2 = 1.2 |

실제로는 학습한 가중치로 Gate 값을 계산한다.

### 4-3. 주요 함수

```python
nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
        dropout=0.0, bidirectional=False, proj_size=0)
output, (h_n, c_n) = lstm(input, hx=None)
```

- `hx=(h_0, c_0)`, 생략하면 둘 다 0.
- `proj_size=0`(기본)이면 h_n, c_n 모두 `[L×D,B,H]`, output은 `[B,T,D×H]`.
- `proj_size`는 출력 Hidden 크기를 투영할 때 사용한다.

### 4-4. 출력 패턴

- **Sequence-to-one**: 마지막 Hidden state로 전체 Sequence의 클래스·값을 예측한다.
- **Sequence-to-sequence**: 각 시점 output으로 태그·복원·다음 값을 예측한다.
- 마지막 index를 쓸 때는 실제 길이와 Padding 위치를 구분한다.
- 양방향은 주어진 Window **내부**를 양쪽으로 읽는다. 예측 시점 이후의 미래 관측이 입력에 섞였는지 확인한다.

### 4-5. LSTM 시계열 예측의 Shape 추적

| 단계 | 연산 | 입력 Shape | 출력 Shape |
|------|------|------------|------------|
| Window | 과거 12개 값 | `[B,12,1]` | `[B,12,1]` |
| LSTM | hidden=16 | `[B,12,1]` | `[B,12,16]` |
| Final | 마지막 시점 | `[B,12,16]` | `[B,16]` |
| Head | Linear 16→1 | `[B,16]` | `[B,1]` |
| Loss | MSE | 예측·정답 `[B,1]` | 스칼라 |
| Batch | 반복 | 샘플 수 가중 평균 | epoch loss |

```python
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.head = nn.Linear(16, 1)
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1])   # 고정 길이 예제 · Padding 없음
```

### 4-6. Sliding Window: `Tensor.unfold`

```python
tensor.unfold(dimension, size, step)
```

- 세 인자 모두 필수. `dimension`: Window를 만들 축, `size`: 한 Window의 길이, `step`: 시작점 간격.
- Window는 원본과 메모리를 공유하는 **view**다.
- 시간 구간을 먼저 나눈 뒤 각각 Window를 생성한다.
- 예: 길이 20, `size=5`, `step=1` → Window 16개.

### 4-7. LSTM에서 자주 발생하는 오류

- h_n과 c_n을 하나의 Tensor로 취급하거나 Shape 순서를 잘못 해석한다.
- Padding된 마지막 열을 실제 마지막 시점으로 사용한다.
- bidirectional 출력의 두 방향을 합치는 방법을 설명하지 않는다.
- 시계열 분할에서 미래 데이터를 train에 섞어 **누수**가 발생한다.

---

## 5. Autoencoder

### 5-1. 개념

입력을 압축하고 다시 복원한다. **정답이 입력 자체**인 재구성 학습을 사용한다.

- **Encoder**: 입력을 더 작은 Latent 표현 z로 압축한다.
- **Latent**: 입력의 주요 패턴을 제한된 차원에 담는다.
- **Decoder**: z로부터 원래 입력을 재구성한다.
- 재구성 오류는 압축 정보의 손실과 입력의 비정상성을 함께 반영한다.

오늘의 구조 (8→4→2→4→8):

| 단계 | 입력 x | 은닉층 | Latent z | 은닉층 | 복원 x̂ |
|------|--------|--------|----------|--------|---------|
| 차원 | 8 | 4 | 2 | 4 | 8 |
| Shape | `[B,8]` | `[B,4]` | `[B,2]` | `[B,4]` | `[B,8]` |

학습 정답은 원래 입력 x이고, 입력 x와 복원 x̂의 MSE를 계산한다.

### 5-2. 설계 시 고려 사항

- 입력 차원보다 작은 **Bottleneck**을 두어 단순 복사를 제한한다.
- Latent 차원이 너무 작으면 중요한 정보도 잃어 **Underfitting**이 발생한다.
- 너무 큰 모델은 입력을 거의 그대로 복사해 일반화가 약해질 수 있다.
- 분류 모델이 아니므로 Latent의 의미는 별도 분석이 필요하다.

### 5-3. 코드와 재구성 오류

```python
model = nn.Sequential(
    nn.Linear(8, 4), nn.ReLU(),
    nn.Linear(4, 2), nn.ReLU(),
    nn.Linear(2, 4), nn.ReLU(),
    nn.Linear(4, 8))
x = torch.randn(5, 8)
recon = model(x)
loss = nn.MSELoss()(recon, x)                  # Batch 평균
score = ((recon - x) ** 2).mean(dim=1)         # 샘플별 오류 [B]
```

- 입력과 재구성 Shape가 일치해야 한다.
- 학습 target은 x 자체다.

### 5-4. 주요 함수

| 함수 | 핵심 |
|------|------|
| `nn.MSELoss(reduction="mean")` | `loss_fn(input, target)`. input은 재구성 결과, target은 원본 입력(같은 Shape). `reduction`은 none/mean/sum(기본 mean). `size_average`·`reduce`는 이전 호환용. |
| `nn.MSELoss(reduction="none")` | 요소별 오류를 유지. `.mean(dim=1)`로 Feature 축을 평균해 `[B,F]` → `[B]`, **샘플별 reconstruction error**를 얻는다. |
| `TensorDataset(*tensors)` | 첫 차원 길이가 같은 Tensor들을 묶는다. |
| `DataLoader(dataset, batch_size=1, shuffle=False, ...)` | `batch_size`·`shuffle`·`num_workers`·`pin_memory`·`drop_last`·`generator`·`persistent_workers`. Autoencoder Batch는 `(x,)` 한 개 Tuple로 반환될 수 있다. |
| `nn.Sequential(*modules)` | 순서대로 실행할 Module 목록. 중간 출력을 자동 반환하지 않는다. |
| `nn.Linear(in_features, out_features, bias=True)` | 입력 마지막 차원을 `in_features`에서 `out_features`로 변환. |

### 5-5. Latent 표현의 활용과 한계

- **차원 축소**: 입력 → Encoder → Latent → Decoder → 재구성.
- **Denoising**: 노이즈 입력 → 깨끗한 target. 정답과 입력의 역할을 구분한다.
- **이상 탐지**: 큰 오류는 이상 **후보**이며 확정이 아니다. 정상 train으로 학습하고, valid에서 임계값을 선택하며, test는 계획 확정 후 선택 과제에서만 평가한다.

### 5-6. Autoencoder도 학습·선택·평가를 분리한다

| train | valid / test |
|-------|--------------|
| 정상 패턴을 학습할 자료를 정의한다. | 같은 전처리 규칙으로 재구성 오류를 계산한다. |
| Normalize·Scaler는 train에서 fit한다. | 임계값은 valid에서 정하고 test는 최종 평가에 사용한다. |
| 노이즈를 추가한다면 입력만 변형하고 target은 깨끗한 원본을 사용한다. | 오류 분포와 실제 이상·정상 사례를 함께 확인한다. |
| 이상 자료를 train에 섞을지 정책을 명시한다. | 모델 선택 후 test 반복 조회를 금지한다. |

### 5-7. 예측(LSTM)과 재구성(Autoencoder)은 목표가 다르다

- **LSTM Forecasting**: 과거 Window로 미래 값 또는 클래스를 예측한다. 미래 정답과의 오차를 평가한다.
- **Autoencoder**: 현재 입력을 압축하고 같은 입력을 복원한다. 재구성 오차를 평가한다.
- 두 결과를 같은 지표로 평균내지 않고 **목적별로 별도 보고**한다.

---

## 6. Attention과 Transformer Encoder

### 6-1. Transformer Encoder의 한 Batch 흐름

1. Token·Feature 입력
2. Embedding + Position
3. Self-Attention
4. FFN·잔차·정규화
5. Pooling·Head

미니 분류 모델의 Shape: 입력 + Position `[B,T,16]` → Transformer Encoder `[B,T,16]` → 실제 위치만 평균 Pooling `[B,16]` → 분류 Head logits `[B,2]`.

- Encoder Layer = Self-Attention + FFN·잔차 연결·LayerNorm.
- 순차 전달 대신 위치 간 관계를 **가중합**으로 계산한다.

### 6-2. Query · Key · Value

- **Query**: 현재 위치가 어떤 정보를 찾는지 나타내는 벡터.
- **Key**: 각 위치가 어떤 정보를 가지고 있는지 비교할 벡터.
- **Value**: Attention 가중치로 실제 합산할 내용 벡터.

$$
\text{Attention}(Q,K,V) = \text{Softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$

가중합 예시 (원리 설명용 숫자, 실제 가중치는 Q·K에서 계산):

| 위치 | 1 | 2 | 3 |
|------|--:|--:|--:|
| Attention 가중치 | 0.2 | 0.3 | 0.5 |
| Value의 한 성분 | 10 | 20 | 30 |

$0.2 \times 10 + 0.3 \times 20 + 0.5 \times 30 = 23$

- **Self-Attention**에서는 query·key·value 모두 같은 입력 표현 x를 각각 선형 투영해 사용한다.

### 6-3. Position · Residual · LayerNorm

- Attention 자체는 입력 순서를 자동으로 구분하지 않는다. Token/Feature Embedding에 **Position Encoding·Embedding**을 더한다.
- **Residual 연결**은 입력과 변환 결과를 더해 Gradient 경로를 돕는다.
- **LayerNorm**은 각 위치의 Feature 차원 분포를 정규화한다.
- Encoder는 동일 차원의 표현을 여러 층 쌓아 문맥을 강화한다.
- 학습 가능한 Position Embedding: `nn.Embedding(max_len, d_model)`. 입력 길이 T ≤ max_len이며, Position Shape `[T,E]` → `[1,T,E]`로 Batch에 Broadcasting해 더한다.

### 6-4. Mask

| 종류 | 설명 |
|------|------|
| Padding Mask `[B,T]` | Attention에서 Padding key를 제외한다. |
| Causal Mask | 미래 위치를 보지 못하게 한다. 다음 토큰 예측에 사용한다. |

- 인자 이름: MHA는 `key_padding_mask`·`attn_mask`, Encoder는 `src_key_padding_mask`·`mask`.
- `nn.MultiheadAttention`의 bool **True는 차단**이다. 코드에서도 True 위치가 Padding이다.
- padded **query**는 집계(Pooling)에서 따로 제외한다.
- Padding Mask는 Encoder에 전달한다.

### 6-5. 주요 함수

```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                      add_zero_attn=False, kdim=None, vdim=None, batch_first=False)

nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                           activation="relu", layer_norm_eps=1e-5,
                           batch_first=False, norm_first=False, bias=True)

nn.TransformerEncoder(encoder_layer, num_layers, norm=None,
                      enable_nested_tensor=True, mask_check=True)
# forward(src, mask=None, src_key_padding_mask=None, is_causal=None)
```

| 함수 | 필수 | 핵심 |
|------|------|------|
| `nn.MultiheadAttention` | `embed_dim`, `num_heads` | `embed_dim`은 `num_heads`로 나누어져야 한다. forward는 `attn_output, attn_output_weights`를 반환. |
| `nn.TransformerEncoderLayer` | `d_model`, `nhead` | `d_model`은 `nhead`로 나누어져야 하며 입력·출력 차원을 유지한다. |
| `nn.TransformerEncoder` | `encoder_layer`, `num_layers` | Layer를 N개 쌓는다. `norm`은 최종 LayerNorm. 출력 Shape는 입력과 같다. |

```python
x = torch.randn(2, 5, 16)
attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
output, weights = attn(x, x, x)    # Self-Attention은 query=key=value=x
# output [B,T,E] → 문맥이 섞인 표현 / weights [B,T,T] → 평균 Head Attention
# need_weights=False면 가중치 반환 생략 가능
```

- `TransformerEncoderLayer` 적용 시 입력·출력 Shape 유지: `[B,T,E]`.

### 6-6. Padding Mask를 포함한 평균 Pooling

```python
def masked_mean(x, padding_mask):
    if bool(padding_mask.all(dim=1).any()):
        raise ValueError("실제 길이는 1 이상")
    valid = (~padding_mask).unsqueeze(-1)
    summed = x.masked_fill(~valid, 0).sum(dim=1)
    return summed / valid.sum(dim=1)
```

- True 위치는 Padding. Pooling 분모는 **실제 길이**.
- 길이 0 입력 정책을 명시한다. 출력은 `[B,E]` → Head.

### 6-7. Transformer의 특징

- Embedding을 Q·K·V로 투영하고 여러 Head가 서로 다른 관계를 본다.
- Attention 결과를 합친 뒤 Residual·Norm·Feedforward를 적용한다.
- 입력 위치를 직접 비교하지만 길이 T가 커지면 Attention 비용이 증가한다.
- Reference 구현은 개념 학습용이며, 최신 대규모 모델은 추가 최적화를 사용한다.

---

## 7. 아키텍처 비교

### 7-1. RNN·LSTM·Autoencoder·Transformer 비교

| 항목 | RNN / LSTM | MLP형 Autoencoder | Transformer |
|------|------------|-------------------|-------------|
| 핵심 연결 | Hidden·Gate 반복 | Encoder–Latent–Decoder | Self-Attention |
| 주 목표 | 분류·예측 | 압축·재구성 | 문맥 분류·예측 |
| 병렬화 | 시간축 순차 | 층 내부 병렬 | 높음 |
| 길이 비용 | O(T) | 입력 차원 의존 | Attention O(T²) |
| 핵심 표현 | Hidden·Cell state | Latent z | 위치별 Embedding |
| 대표 지표 | Accuracy·MSE | Reconstruction error | Accuracy·AUC·MSE |

- 점수 하나가 아니라 **입력 구조와 학습 비용**을 함께 비교한다.
- RNN/LSTM이 시간축에서 순차적인 이유: 현재 상태가 이전 상태에 의존하기 때문이다.
- Transformer 비용이 길이에 따라 크게 늘어나는 이유: 모든 위치 쌍의 점수를 계산해 T×T 관계를 만들기 때문이다.

### 7-2. 공정하게 비교하는 방법

성능표에는 점수와 함께 **고정한 조건**과 **바꾼 항목**을 남긴다.

| 비교 목적 | 고정하는 조건 | 바꾸는 항목 |
|-----------|---------------|-------------|
| 모델 비교 | 자료·분할·목표·Device | RNN / LSTM / Transformer |
| 장치 비교 | 모델·가중치·입력·측정 구간 | CPU / GPU |
| 재구성 실험 | 입력·분할·전처리 | Latent 크기 등 |

- 예측은 예측끼리, 분류는 분류끼리 같은 지표로 비교한다. Autoencoder의 재구성 결과는 별도로 기록한다.
- 같은 train·valid·test·Window·Feature·Target을 사용한다.
- Epoch·Optimizer·주 지표·선택 규칙을 먼저 정한다.
- 파라미터 수·학습 시간·Peak Memory와 성능을 함께 기록한다.

### 7-3. 아키텍처 비교표에 반드시 남길 항목

| 모델 | 주 목표 | Params | Best Epoch | Valid 지표 | 시간·메모리 |
|------|---------|--------|------------|------------|-------------|
| RNN | 분류/예측 | 측정 | 측정 | 측정 | 측정 |
| LSTM | 분류/예측 | 측정 | 측정 | 측정 | 측정 |
| Transformer | 분류/예측 | 측정 | 측정 | 측정 | 측정 |

### 7-4. Sequence 모델에서 자주 발생하는 오류

- BTF와 TBF 축 순서를 혼동해 Batch·Time이 뒤바뀐다.
- Padding 위치를 마지막 상태·평균 Pooling에 포함한다.
- 시점별 예측에서 허용하지 않은 미래 관측을 참조한다.
- 서로 다른 목표의 지표·데이터 분할을 한 표에서 직접 순위화한다.

---

## 8. 길이·Padding·Device 성능 비교

### 8-1. 공정한 Sequence 모델 비교 조건

- 같은 Sequence 길이 분포·Batch size·dtype·Device·Warm-up을 사용한다.
- 데이터 로딩·Padding 생성 시간을 포함할지 측정 범위를 명시한다.
- GPU는 비동기 실행이므로 시간 전후 `synchronize`를 적용한다.
- 평균 시간·samples/s·Peak VRAM·환경·Padding 비율을 함께 기록한다.

### 8-2. GPU 시간·Peak Memory 측정 함수

```python
torch.cuda.synchronize(device=None)
torch.cuda.reset_peak_memory_stats(device=None)
torch.cuda.max_memory_allocated(device=None)
```

- 필수 인자 없음. `device`는 측정할 CUDA Device이며 기본은 current device.
- CUDA 사용 가능 환경에서 호출한다. **CPU에서는 호출하지 않고 N/A로 기록**한다.
- 반환: `synchronize`·`reset`은 없음, `max_memory_allocated`는 byte.
- 측정 범위·Warm-up·반복 횟수를 함께 기록한다.

### 8-3. 한 Batch의 순전파를 반복해 시간 측정

```python
import time, torch
# model(x, lengths) -> logits; 입력·모델 장치는 동일
# 호출 전 model.eval(); 전송 시간은 제외한 예제
with torch.inference_mode():
    for _ in range(2):            # 워밍업
        model(x, lengths)
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    start = time.perf_counter()
    for _ in range(20):
        model(x, lengths)
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    seconds = time.perf_counter() - start
print("samples/s:", len(x) * 20 / seconds)
```

- 한 epoch가 아닌 **20회 순전파**를 측정한다.
- `samples/s = B × 20 / seconds`.
- CPU의 VRAM은 N/A, 전송 시간은 제외한다.
- 전체 상태 보존은 특심화07에서 다룬다.

### 8-4. Sequence 길이와 속도·메모리의 관계

| 길이 T | RNN/LSTM 시간 | Transformer 시간 | Peak VRAM | 관찰 |
|-------:|:-------------:|:----------------:|:---------:|------|
| 16 | 측정 | 측정 | 측정 | 짧은 기준 |
| 32 | 측정 | 측정 | 측정 | 비용 증가 |
| 64 | 측정 | 측정 | 측정 | Attention T² 영향 |
| 128 | 측정 | 측정 | 측정 | 메모리 한도 확인 |

같은 Device에서 모델별로 비교하는 결과표:

| Device | Model | T | Params | Valid | 시간 | Peak VRAM |
|--------|-------|---|--------|-------|------|-----------|
| CPU | RNN | 기록 | 기록 | 기록 | 기록 | N/A |
| CPU | LSTM | 기록 | 기록 | 기록 | 기록 | N/A |
| CPU | Transformer | 기록 | 기록 | 기록 | 기록 | N/A |

GPU는 같은 표로 별도 측정하며, 장치 비교는 같은 모델을 유지한다.

### 8-5. 성능 측정값을 과장하지 않고 해석한다

- GPU 이점은 모델 크기·길이·Batch·전송 비용·커널 최적화에 따라 달라진다.
- Transformer가 항상 빠르거나 RNN이 항상 가벼운 것은 아니다.
- 작은 입력에서는 Launch·전송 오버헤드가 계산 시간을 지배할 수 있다.
- 성능표에는 하드웨어·버전·dtype·Mask·측정 범위를 함께 남긴다.

---

## 9. 프로젝트 구성과 제출 산출물

### 9-1. 최종 프로젝트: 시계열 예측과 재구성 미니 모델

- LSTM으로 과거 Window에서 다음 시점 값을 예측한다.
- Autoencoder로 입력을 압축·복원하고 재구성 오류를 기록한다.
- RNN·LSTM·Transformer 비교는 같은 분할·지표·길이에서 수행한다.
- **기본필수**는 한 파일 결과와 README 설명, 상위 단계는 모델 복원·검증까지 제출한다.

프로젝트 흐름: ① 자료·Window 확인 → ② 길이 확인·가변 길이 Mask → ③ LSTM·AE 학습 → ④ 예측·재구성 결과 기록 → ⑤ 성능표·보고서 → ⑥ 상위 단계 저장·복원

### 9-2. 심화 참고: 모듈형 프로젝트 구조

```text
day12_sequence_models/
├── main.py        # 실험 실행과 결과 저장
├── models.py      # RNN·LSTM·Autoencoder·Transformer 정의
├── data.py        # Window·Padding·Mask·DataLoader 구성
├── train.py       # train/evaluate·checkpoint·benchmark 함수
├── outputs/       # history.csv·comparison.csv·predictions.csv·best_model.pt
└── reports/
    └── architecture_comparison.md   # 선택 근거·오류 사례·한계
```

### 9-3. 단계별 제출물과 기록 내용

| 산출물 | 기록 내용 |
|--------|-----------|
| `history.csv` | epoch·train/valid loss·주 지표·best 표시 |
| `architecture_comparison.csv` | 모델·Params·길이·Valid·시간·메모리 |
| `predictions.csv` | ID·실제값·예측값 또는 재구성 오류 |
| `best_model.pt` | state_dict·모델 설정·Feature·길이 정보 |
| `mask_or_latent.png` | Padding Mask 또는 Latent·재구성 시각화 |
| `README/report` | 실행법·선택 근거·오류 사례·한계·재현 조건 |

기본필수는 자기 폴더 README 기준이며, 저장·복원 등은 해당 상위 단계 기준이다.

### 9-4. 형성평가 · 기본필수만으로 100점

| 평가 항목 | 배점 | 충족 기준 |
|-----------|-----:|-----------|
| Sequence·Mask | 20 | BTF Shape·Padding·실제 길이·분할 확인 |
| 모델 구현 | 25 | RNN/LSTM/AE/Transformer 중 목표에 맞는 구조·Shape 설명 |
| 학습·평가 | 25 | 자기 단계의 지표·원문 결과·test 보관 설명 |
| 비교·성능 | 15 | 측정 조건·모델별 목적·CPU 환경의 한계 |
| 재현·설명 | 15 | 실행법·환경·결과 재생성·개인 설명 |

### 9-5. 정상·경계·실패 상황 테스트

| 상황 | 기대 결과 | 판정 근거 |
|------|-----------|-----------|
| 정상 Sequence Batch | `[B,T,F]` → 목표 Shape | loss·예측 개수 일치 |
| 가변 길이·Padding | Mask로 채움 위치 제외 | Pooling·Attention 분모 확인 |
| 길이 1 Sequence | 모델이 명확히 처리 | 마지막 상태·Mask Shape 확인 |
| 모든 위치 Padding | 중단 또는 정책 적용 | 0으로 나누지 않음 |
| 저장 모델 복원 | 동일 입력의 출력 대조 | 구조·설정·Feature 일치 |
| CPU-only 환경 | 기본 실습 정상, GPU N/A | device 분기·보고서 기록 |

### 9-6. 실습 단계별 과제 (기본필수 중심)

| 실습 | 시간 | 기본필수 |
|------|-----:|----------|
| 01 Sequence Tensor와 Padding | 20분 | 길이 4·2·5인 Tensor 3개를 Padding해 `[3,5,3]` Shape 확인 |
| 02 RNN 분류와 시점 출력 | 25분 | `[B,6,3]` 입력을 RNN에 넣고 output·h_n Shape와 분류 Logit 확인 |
| 03 LSTM 시계열 예측 | 25분 | 사인파 Window `[B,12,1]`로 다음 값 하나를 예측하고 Shape 설명 |
| 04 Autoencoder 재구성 | 25분 | 8차원 정상 데이터를 2차원 Latent로 압축하고 재구성 MSE 기록 |
| 05 Self-Attention과 Transformer Encoder | 25분 | `[B,5,16]` 입력으로 Attention output·weight Shape 확인 |
| 06 아키텍처 비교와 Mask 처리 | 25분 | 같은 Sequence 분류 자료에서 RNN·LSTM의 Params·valid Accuracy 비교 |
| 07 Sequence 모델 시간·메모리 비교 | 25분 | RNN 한 Batch의 CPU 시간과 samples/s 측정, 조건 기록 |
| 08 아키텍처 비교 프로젝트 통합·시연 | 30분 | 제공 골격에서 LSTM 예측·Autoencoder 재구성·결과표 저장 완성 |

각 실습은 기본필수 → 응용 → 심화 → 특심화의 4단계로 구성된다.

---

## 10. 핵심 개념 빠른 복습

- **Sequence Tensor**: 순서가 있는 입력. 기본 Shape는 `[Batch, Time, Feature]`.
- **Window**: 과거 T개 시점을 묶은 입력 단위. 이번 실습은 T=12, F=1.
- **Padding**: 짧은 Sequence를 채워 Batch 안의 최대 길이 T로 맞추는 것.
- **lengths**: 샘플별 실제 길이 `[B]`.
- **Mask**: 실제 값과 Padding 위치를 구분하는 표시. True가 Padding(차단).
- **PackedSequence**: Padding 없이 실제 길이만 담는 표현. RNN/LSTM 입력·출력으로 사용 가능.
- **Hidden state (h)**: 현재까지 읽은 정보를 고정된 `hidden_size`로 요약한 상태.
- **Cell state (c)**: LSTM이 장기 정보를 비교적 직접적인 경로로 전달하는 상태.
- **Gate**: 기억을 지울지(Forget)·쓸지(Input)·출력할지(Output) 0~1로 조절하는 장치.
- **Many-to-one / Many-to-many / Next-step / Encoder–Decoder**: Sequence 입출력의 대표 구조.
- **output vs h_n**: output은 모든 시점의 마지막 층 출력, h_n은 모든 층·방향의 마지막 상태.
- **Autoencoder**: 입력을 압축(Encoder)한 Latent에서 다시 복원(Decoder)하는 재구성 학습 구성.
- **Latent z**: 입력의 주요 패턴을 담은 작은 표현.
- **Bottleneck**: 입력보다 작은 중간 차원으로 단순 복사를 제한하는 구조.
- **Reconstruction error**: 입력과 복원 결과의 오차. 큰 값은 이상 *후보*이지 확정이 아님.
- **Denoising Autoencoder**: 노이즈 입력 → 깨끗한 원본 target.
- **Query / Key / Value**: 찾을 정보 / 비교할 위치 / 합산할 내용.
- **Self-Attention**: query=key=value=x(각각 선형 투영)로 같은 입력 안의 위치끼리 비교.
- **Multi-head**: 여러 Head가 서로 다른 관계를 병렬로 본다.
- **Position Encoding / Embedding**: Attention에 순서 정보를 넣기 위해 더하는 표현.
- **Residual 연결 / LayerNorm / FFN**: Gradient 경로 보조 / Feature 차원 정규화 / Layer 안의 Feedforward Network.
- **Padding Mask vs Causal Mask**: Padding key 제외 / 미래 위치 차단.
- **Warm-up**: 측정 전에 몇 번 미리 실행해 두는 것.
- **samples/s**: 초당 처리한 샘플 수 (`B × 반복 횟수 / seconds`).
- **Peak VRAM**: 측정 구간의 최대 GPU 메모리 (`max_memory_allocated`, byte). CPU는 N/A.
- **누수(Leakage)**: 미래 데이터나 test 정보가 train·모델 선택에 섞이는 오류.

---

## 11. 수업 퀴즈 핵심 정답

**CNN → Sequence 모델 (복습)**

1. 시계열의 순서를 임의로 섞어도 같은 문제인가? → 아니다. 순서 자체가 입력 정보이므로 셔플 위치와 단위를 구분해야 한다.
2. 모델 구조가 바뀌어도 train·valid·test 원칙이 필요한가? → 필요하다. 모델 선택과 최종 평가의 분리는 그대로 유지한다.
3. 입력 길이가 서로 다르면 Batch를 위해 무엇이 필요한가? → Padding과 길이 또는 Mask. 실제 토큰과 채움 위치를 구분한다.

**RNN**

4. `batch_first=True`일 때 RNN 입력 Shape는? → `[Batch, Time, Feature]`.
5. output과 h_n은 같은 Tensor인가? → 아니다. output은 모든 시점, h_n은 마지막 상태다.
6. 기본 RNN이 긴 Sequence에서 어려운 이유는? → 긴 시간축의 Gradient와 정보 보존이 어렵고 순차 계산이 필요하다.

**LSTM**

7. LSTM이 추가로 유지하는 상태는? → Cell state c. Hidden state h와 함께 반환된다.
8. `output[:, -1]`을 항상 실제 마지막 시점으로 볼 수 있는가? → Padding이 있으면 아니다. 실제 lengths 또는 packed 입력을 사용한다.
9. 양방향 LSTM을 미래 예측에 쓸 때 확인할 점은? → 예측 시점 이후 관측을 넣지 않았는지 확인한다. 과거 Window 안의 역방향 읽기 자체는 누수가 아니다.

**Autoencoder**

10. 기본 Autoencoder의 target은? → 입력 자체. Denoising은 깨끗한 원본이 target이다.
11. 재구성 오류가 크면 반드시 이상인가? → 아니다. 모델 용량·전처리·학습 분포의 영향도 받는다.
12. 이상 탐지 임계값을 test를 보며 골라도 되는가? → 아니다. valid에서 고정하고 test는 최종 평가에 사용한다.

**Transformer**

13. Self-Attention에서 query·key·value는 무엇을 사용하나? → 같은 입력 표현 x를 각각 선형 투영해 사용한다.
14. Transformer에 Position 정보가 필요한 이유는? → Attention 자체는 입력 순서를 자동으로 구분하지 않기 때문이다.
15. Padding 위치를 무시하기 위해 쓰는 Mask는? → key padding mask. Batch별 Padding 위치를 표시한다.

**아키텍처 비교**

16. RNN·LSTM의 계산이 시간축에서 순차적인 이유는? → 현재 상태가 이전 상태에 의존하기 때문이다.
17. Transformer Attention 비용이 길이에 따라 크게 늘어나는 이유는? → 모든 위치 쌍의 점수를 계산해 T×T 관계를 만들기 때문이다.
18. Autoencoder와 분류 모델의 지표를 직접 평균내도 되는가? → 아니다. 목표와 단위가 다른 재구성·분류 지표다.

**Sequence 성능 측정**

19. GPU 시간을 재기 전에 `synchronize`가 필요한 이유는? → CUDA 실행이 비동기이므로 실제 완료 시점을 맞추기 위해서다.
20. Padding 비율이 높으면 어떤 문제가 생기는가? → 실제 정보가 아닌 위치에도 연산·메모리를 사용한다.
21. 길이를 바꾸며 모델을 비교할 때 고정해야 할 것은? → 데이터·Batch·dtype·Device·측정 범위·모델 설정.

---

## 12. 오늘의 핵심 정리와 학습 범위 메모

- RNN은 Hidden state, LSTM은 Gate와 Cell state로 순서 정보를 처리한다.
- Autoencoder는 압축·재구성하며 재구성 오류를 별도의 목표로 해석한다.
- Transformer는 Attention·Position·Mask로 입력 위치 관계를 직접 계산한다.
- 아키텍처 비교는 데이터·지표·길이·시간·메모리·재현 조건을 함께 기록한다.

이 README는 제공된 수업 자료(`20261102_2.pdf`)의 내용을 학습용으로 재구성한 것이다. 자료에서 다루지 않은 세부 구현이나 외부 이론은 임의로 보충하지 않았다.

다음 수업(자료 기준 2026-11-04) 예고: **Vision·OpenCV**

- OpenCV 이미지 표현: NumPy 배열과 BGR·RGB 색공간, Resize·Blur·Edge·형태학 처리, 이미지·영상 입출력과 Frame 반복
- 필터·영상·증강 파이프라인: 원본 대비 전처리 결과 검증, CNN 전처리와 연결, Vision 프로젝트의 입력 품질 관리

공식 참고자료: PyTorch의 `nn.RNN`/`RNNCell`, `nn.LSTM`/`LSTMCell`, `pad_sequence`·`pack_padded_sequence`·`pad_packed_sequence`, `TensorDataset`/`DataLoader`/`MSELoss`, `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`/`TransformerEncoder`, Transformer tutorial and masking examples, *Attention Is All You Need* (원 논문).
