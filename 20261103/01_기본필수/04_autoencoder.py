"""
[Autoencoder (오토인코더)]

1. Autoencoder란?
- 입력 데이터를 압축한 다음 다시 원래 형태로 복원하는 신경망이다.
- 입력 데이터와 복원 데이터가 최대한 비슷해지도록 학습한다.
- 별도의 정답 데이터 대신 입력 데이터 자체를 정답으로 사용한다.

2. Encoder (인코더)
- 입력 데이터를 압축하는 역할을 담당한다.
- 입력 데이터에서 중요한 특징을 추출한다.
- 이번 실습에서는 8개의 데이터를 2개의 데이터로 압축한다.

3. Decoder (디코더)
- 압축된 데이터를 다시 원래 형태로 복원하는 역할을 담당한다.
- 이번 실습에서는 2개의 데이터를 다시 8개의 데이터로 복원한다.

4. Latent (잠재 표현)
- Encoder가 입력 데이터를 압축하여 만들어 낸 표현이다.
- 원본 데이터보다 작은 차원으로 데이터를 표현한다.
- 이번 실습에서는 8개 데이터가 2개 데이터로 압축된 결과를 의미한다.

5. Reconstruction (복원)
- Decoder가 압축된 데이터를 이용해 다시 만들어 낸 결과다.
- 입력 데이터와 복원 데이터가 얼마나 비슷한지가 중요하다.

6. Reconstruction Error (복원 오차)
- 입력 데이터와 복원 데이터 사이의 차이를 의미한다.
- 오차가 작을수록 입력 데이터를 잘 복원했다고 볼 수 있다.

7. MSE (Mean Squared Error)
- 예측값과 실제값의 차이를 제곱한 다음 평균을 계산하는 손실 함수다.
- Autoencoder에서는 복원 데이터와 원본 입력 데이터의 차이를 계산한다.
- 손실값이 작아지도록 모델을 학습한다.

8. Tanh (하이퍼볼릭 탄젠트)
- 신경망에서 사용하는 활성화 함수 중 하나다.
- 입력값을 -1부터 1 사이의 값으로 변환한다.
- 비선형적인 관계를 학습할 수 있도록 돕는다.

9. Sequential
- 여러 신경망 계층을 순서대로 연결하는 구조다.
- 앞 계층의 출력이 다음 계층의 입력으로 전달된다.

10. Anomaly Detection (이상 탐지)
- 정상적인 데이터와 다른 특이한 데이터를 찾아내는 작업이다.
- 정상 데이터로 학습한 Autoencoder는 정상 패턴을 비교적 잘 복원할 수 있다.
- 학습 과정에서 보지 못한 이상 데이터는 복원 오차가 커질 수 있다.
- 복원 오차를 기준으로 이상 여부를 판단하는 데 활용할 수 있다.

[이번 실습의 핵심 구조]

입력 데이터 8개
    ↓
Encoder
    ↓
잠재 표현 2개 (Latent)
    ↓
Decoder
    ↓
복원 데이터 8개 (Reconstruction)
    ↓
원본 입력 데이터와 비교 (MSE)

[핵심 정리]
- Autoencoder는 입력 데이터를 압축하고 복원하는 신경망이다.
- Encoder는 압축, Decoder는 복원을 담당한다.
- Latent는 압축된 데이터 표현이다.
- 입력 데이터 자체가 학습의 정답이 된다.
- 복원 오차를 줄이는 방향으로 학습한다.
- 정상 데이터와 이상 데이터의 복원 오차 차이를 이용해 이상 탐지에 활용할 수 있다.
"""
import torch
from torch import nn

# 작은 CPU 실습에서 사용할 스레드 수를 1개로 고정한다.
torch.set_num_threads(1)

# 난수 시드를 고정하여 실행 결과의 재현성을 높인다.
torch.manual_seed(42)

from pathlib import Path
import pandas as pd

# 현재 Python 파일이 위치한 폴더를 ROOT로 지정한다.
ROOT = Path(__file__).resolve().parent

# ROOT 폴더에 있는 series.csv 파일을 읽어 데이터프레임으로 저장한다.
df = pd.read_csv(ROOT / "series.csv")


# 시간 구간을 먼저 분리한다.
# train은 모델 학습에 사용하고 valid는 학습된 모델의 성능 검증에 사용한다.
# test 구간은 이번 코드에서 사용하지 않는다.

# split 열의 값이 train인 행만 선택한다.
# value 열의 데이터를 가져와 PyTorch 텐서로 변환한다.
# float32는 신경망 학습에서 사용하는 실수형 자료형이다.
train = torch.tensor(
    df.loc[df["split"] == "train", "value"].values,
    dtype=torch.float32
)

# split 열의 값이 valid인 행만 선택한다.
# 검증 데이터도 동일한 방식으로 텐서로 변환한다.
valid = torch.tensor(
    df.loc[df["split"] == "valid", "value"].values,
    dtype=torch.float32
)


# Sliding Window 방식으로 시계열 데이터를 일정한 크기로 분리한다.
#
# unfold(0, 8, 1)
# - 0: 시간 방향(첫 번째 차원)을 기준으로 분리한다.
# - 8: 한 번에 8개의 데이터를 묶는다.
# - 1: Window를 한 칸씩 이동한다.
#
# 예시:
# [1, 2, 3, 4, 5, 6, 7, 8]
#    [2, 3, 4, 5, 6, 7, 8, 9]
#       [3, 4, 5, 6, 7, 8, 9, 10]
#
# 이번 실습은 다음 값을 예측하는 것이 아니라
# 각 Window에 포함된 8개 데이터 자체를 압축하고 복원한다.
#
# contiguous()
# 텐서의 메모리 배치를 연속적인 형태로 정리한다.
x = train.unfold(0, 8, 1).contiguous()

# 검증 데이터도 동일한 방식으로 Window를 생성한다.
v = valid.unfold(0, 8, 1).contiguous()


# Autoencoder 모델을 생성한다.
#
# nn.Sequential:
# 여러 신경망 계층을 순서대로 연결한다.
# 앞 계층의 출력이 다음 계층의 입력으로 전달된다.
#
# nn.Linear(8, 2):
# Encoder 역할을 담당한다.
# 입력 데이터 8개를 2개의 값으로 압축한다.
#
# nn.Tanh():
# 활성화 함수다.
# 입력값을 -1부터 1 사이의 범위로 변환한다.
# 신경망이 비선형적인 관계도 학습할 수 있도록 돕는다.
#
# nn.Linear(2, 8):
# Decoder 역할을 담당한다.
# 압축된 2개의 값을 다시 8개의 값으로 복원한다.
#
# 전체 구조:
# 입력 8개 → 압축 2개 → 복원 8개
model = nn.Sequential(
    nn.Linear(8, 2),
    nn.Tanh(),
    nn.Linear(2, 8)
)


# Adam 옵티마이저를 생성한다.
#
# model.parameters():
# 모델 내부의 학습 가능한 가중치와 편향을 가져온다.
#
# lr=0.03:
# 학습률(Learning Rate)을 의미한다.
# 학습 과정에서 가중치를 얼마나 크게 업데이트할지 결정한다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)


# 총 40번의 Epoch 동안 모델을 학습한다.
# Epoch는 학습 데이터를 한 번 사용하는 단위다.
for epoch in range(40):

    # 이전 학습에서 계산된 기울기를 초기화한다.
    # 매 학습마다 새로운 기울기를 계산하기 위해 필요하다.
    optimizer.zero_grad()

    # 입력 데이터 x를 Autoencoder에 전달한다.
    #
    # Encoder가 8개 데이터를 2개로 압축하고
    # Decoder가 다시 8개 데이터로 복원한다.
    #
    # reconstructed:
    # 모델이 복원한 데이터
    #
    # x 형태: [Window 개수, 8]
    # reconstructed 형태: [Window 개수, 8]
    reconstructed = model(x)


    # Autoencoder의 정답은 입력 데이터 x 자체다.
    #
    # 일반적인 지도학습에서는 입력과 별도의 정답을 사용하는 경우가 많지만,
    # Autoencoder는 입력 데이터를 다시 복원하는 것이 목적이므로
    # 입력 데이터 자체를 정답으로 사용한다.
    #
    # reconstructed:
    # Autoencoder가 복원한 데이터
    #
    # x:
    # 원본 입력 데이터이자 정답 데이터
    #
    # nn.MSELoss():
    # 복원 데이터와 원본 데이터 사이의 평균 제곱 오차를 계산한다.
    #
    # torch.zeros_like(x)를 사용하면 안 되는 이유:
    # 모든 값을 0으로 만드는 것이 아니라
    # 원래 입력 데이터와 비슷하게 복원하는 것이 목적이기 때문이다.
    loss = nn.MSELoss()(reconstructed, x)


    # 계산된 손실을 바탕으로 각 파라미터의 기울기를 계산한다.
    # 역전파(Backpropagation)를 수행한다.
    loss.backward()

    # 계산된 기울기를 이용해 모델의 가중치와 편향을 업데이트한다.
    optimizer.step()


# 학습이 끝났으므로 모델을 평가 모드로 변경한다.
model.eval()

# 검증 과정에서는 학습을 위한 기울기를 계산하지 않는다.
# 불필요한 연산을 줄이고 예측 및 복원 결과만 확인한다.
with torch.inference_mode():

    # model[:2]
    # Sequential 모델의 앞쪽 두 계층만 가져온다.
    #
    # Linear(8, 2) + Tanh()
    # Encoder 부분만 실행한다.
    #
    # 즉, 검증 데이터 8개를 2개로 압축한다.
    #
    # latent:
    # 압축된 잠재 표현
    #
    # 형태: [검증 Window 개수, 2]
    latent = model[:2](v)

    # 검증 데이터 v를 전체 Autoencoder에 전달한다.
    # 압축한 다음 다시 8개 데이터로 복원한다.
    #
    # nn.MSELoss():
    # 복원 결과와 원본 검증 데이터의 차이를 계산한다.
    #
    # item():
    # 텐서에 담긴 숫자 하나를 Python 숫자로 가져온다.
    error = nn.MSELoss()(model(v), v).item()


# 학습 데이터와 검증 데이터의 형태를 출력한다.
print("train:", tuple(x.shape), "valid:", tuple(v.shape))

# Latent와 복원 결과의 형태를 출력한다.
#
# latent:
# 8개 데이터가 2개로 압축된 결과
#
# model(v):
# 압축된 데이터가 다시 8개로 복원된 결과
print("latent / 복원:", tuple(latent.shape), tuple(model(v).shape))

# 검증 데이터의 복원 오차(MSE)를 출력한다.
# 값이 작을수록 원본 데이터를 잘 복원했다는 의미다.
print("valid MSE:", round(error, 6))