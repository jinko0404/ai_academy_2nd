"""
실습 A04: Autoencoder의 Latent Size 비교

[실습 목적]
- Autoencoder의 압축 크기를 1, 2, 4로 변경한다.
- 압축 크기에 따른 복원 성능을 비교한다.
- 각 모델의 파라미터 개수와 검증 MSE를 확인한다.
- 실험 결과를 CSV 파일로 저장한다.

[핵심 개념]

1. Latent Size
   - Encoder가 입력 데이터를 몇 개의 값으로 압축할지 결정한다.
   - 이번 실습에서는 1, 2, 4를 비교한다.

2. Retained Ratio
   - 원본 데이터 크기 대비 압축된 데이터 크기의 비율이다.
   - latent_size / 8로 계산한다.
   - 실제 정보 보존율을 의미하는 것은 아니다.

3. Params
   - 신경망이 학습하는 가중치와 편향의 총개수다.
   - Latent Size가 커지면 일반적으로 파라미터 수도 증가한다.

4. Best Epoch
   - 학습 과정에서 검증 MSE가 가장 낮았던 Epoch다.

5. Valid MSE
   - 검증 데이터의 원본과 복원 결과 사이의 평균 제곱 오차다.
   - 낮을수록 복원 오차가 작다는 의미다.

6. 실험 비교
   - 압축 크기가 작으면 모델이 표현할 수 있는 정보의 양이 제한될 수 있다.
   - 압축 크기가 크면 더 많은 정보를 표현할 수 있지만 모델의 복잡도도 증가한다.
   - 각 압축 크기의 복원 성능과 파라미터 수를 비교한다.
"""

import torch
from torch import nn

# 작은 CPU 실습에서 사용할 스레드 수를 1개로 고정한다.
torch.set_num_threads(1)

# 난수 시드를 고정하여 실험 결과의 재현성을 높인다.
torch.manual_seed(42)

from pathlib import Path
import pandas as pd

# 현재 Python 파일이 위치한 폴더를 ROOT로 지정한다.
ROOT = Path(__file__).resolve().parent

# 실험 결과를 저장할 outputs 폴더 경로를 지정한다.
OUT = ROOT / "outputs"

# outputs 폴더가 없으면 생성한다.
# exist_ok=True는 폴더가 이미 존재해도 오류가 발생하지 않게 한다.
OUT.mkdir(exist_ok=True)

# data 폴더에 있는 series.csv 파일을 읽어 데이터프레임으로 저장한다.
df = pd.read_csv(ROOT / "data" / "series.csv")


# split 열을 기준으로 학습 데이터와 검증 데이터를 분리한다.

# train에 해당하는 행의 value 데이터를 가져온다.
# PyTorch에서 사용할 수 있도록 float32 텐서로 변환한다.
train = torch.tensor(
    df.loc[df["split"] == "train", "value"].values,
    dtype=torch.float32
)

# valid에 해당하는 행의 value 데이터도 동일한 방식으로 변환한다.
valid = torch.tensor(
    df.loc[df["split"] == "valid", "value"].values,
    dtype=torch.float32
)


# Sliding Window 방식으로 시계열 데이터를 8개씩 분리한다.
#
# unfold(0, 8, 1)
# - 0: 첫 번째 차원을 기준으로 분리한다.
# - 8: 한 번에 8개의 데이터를 묶는다.
# - 1: Window를 한 칸씩 이동한다.
#
# contiguous():
# 텐서의 메모리 배치를 연속적인 형태로 정리한다.
#
# x: 학습용 Window 데이터
# v: 검증용 Window 데이터
#
# rows:
# 각 Latent Size별 실험 결과를 저장할 빈 리스트다.
x = train.unfold(0, 8, 1).contiguous()
v = valid.unfold(0, 8, 1).contiguous()
rows = []


# Latent Size를 1, 2, 4로 변경하면서 실험을 반복한다.
#
# latent_size = 1 → 8개를 1개로 압축
# latent_size = 2 → 8개를 2개로 압축
# latent_size = 4 → 8개를 4개로 압축
for latent_size in [1, 2, 4]:

    # 각 실험을 동일한 난수 조건에서 시작하도록 시드를 고정한다.
    # 모델 초기 가중치 생성에 사용되는 난수를 동일하게 맞춘다.
    torch.manual_seed(42)


    # Autoencoder 모델을 생성한다.
    #
    # nn.Sequential:
    # 여러 신경망 계층을 순서대로 연결한다.
    #
    # nn.Linear(8, latent_size):
    # Encoder 역할을 담당한다.
    # 입력 데이터 8개를 latent_size 개수로 압축한다.
    #
    # nn.Tanh():
    # 활성화 함수다.
    # 입력값을 -1부터 1 사이로 변환하며 비선형성을 추가한다.
    #
    # nn.Linear(latent_size, 8):
    # Decoder 역할을 담당한다.
    # 압축된 데이터를 다시 8개로 복원한다.
    #
    # 반복문에 따라 모델 구조가 자동으로 변경된다.
    # 예: latent_size가 2라면 8 → 2 → 8 구조가 된다.
    model = nn.Sequential(
        nn.Linear(8, latent_size),
        nn.Tanh(),
        nn.Linear(latent_size, 8)
    )


    # Adam 옵티마이저를 생성한다.
    #
    # model.parameters():
    # 모델 내부의 학습 가능한 가중치와 편향을 가져온다.
    #
    # lr=0.02:
    # 학습률을 0.02로 설정한다.
    #
    # best:
    # 지금까지 확인한 가장 낮은 검증 MSE를 저장한다.
    # 처음에는 무한대로 설정하여 첫 번째 검증 결과가 갱신되도록 한다.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    best = float("inf")


    # 총 40 Epoch 동안 모델을 학습한다.
    # range(1, 41)은 1부터 40까지 반복한다.
    for epoch in range(1, 41):

        # 이전 학습에서 계산된 기울기를 초기화한다.
        optimizer.zero_grad()

        # 학습 데이터를 Autoencoder에 전달하여 복원한다.
        #
        # model(x):
        # 입력 8개를 압축한 다음 다시 8개로 복원한다.
        #
        # nn.MSELoss()(model(x), x):
        # 복원 결과와 원본 입력 데이터의 차이를 계산한다.
        # Autoencoder에서는 입력 데이터 x 자체가 정답이다.
        loss = nn.MSELoss()(model(x), x)

        # 손실을 바탕으로 각 파라미터의 기울기를 계산한다.
        # 역전파(Backpropagation)를 수행한다.
        loss.backward()

        # 계산된 기울기를 이용해 모델의 파라미터를 업데이트한다.
        optimizer.step()


        # 검증 단계에서는 기울기를 계산하지 않는다.
        with torch.inference_mode():

            # 검증 데이터 v를 Autoencoder로 복원한다.
            # 복원 결과와 원본 v의 MSE를 계산한다.
            #
            # item():
            # 텐서에 저장된 숫자 하나를 Python 숫자로 가져온다.
            score = nn.MSELoss()(model(v), v).item()


        # 이번 Epoch의 검증 MSE가 기존 최저값보다 낮은지 확인한다.
        if score < best:

            # 기존 최저값보다 낮으면 최저 MSE를 갱신한다.
            best = score

            # 가장 낮은 검증 MSE가 나온 Epoch를 기록한다.
            best_epoch = epoch


    # 현재 Latent Size의 실험 결과를 딕셔너리로 만들어 rows에 추가한다.
    rows.append({

        # 현재 실험에서 사용한 압축 크기
        "latent": latent_size,

        # 원본 데이터 크기 8 대비 압축된 데이터 크기의 비율
        "retained_ratio": latent_size / 8,

        # 모델의 전체 파라미터 개수
        #
        # model.parameters():
        # 모델의 모든 학습 가능한 파라미터를 가져온다.
        #
        # p.numel():
        # 각 파라미터 텐서에 포함된 원소의 개수를 계산한다.
        #
        # sum():
        # 모든 파라미터 원소 개수를 합산한다.
        "params": sum(p.numel() for p in model.parameters()),

        # 검증 MSE가 가장 낮았던 Epoch
        "best_epoch": best_epoch,

        # 학습 중 기록한 가장 낮은 검증 MSE
        "valid_mse": best
    })


# rows에 저장한 실험 결과를 데이터프레임으로 변환한다.
table = pd.DataFrame(rows)

# 실험 결과를 outputs 폴더에 CSV 파일로 저장한다.
#
# index=False:
# 데이터프레임의 행 번호를 별도 열로 저장하지 않는다.
table.to_csv(OUT / "latent_comparison.csv", index=False)

# 실험 결과를 소수점 6자리까지 반올림하여 표 형태로 출력한다.
print(table.round(6).to_string(index=False))