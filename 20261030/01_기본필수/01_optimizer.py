from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
df = pd.read_csv(base / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# 손실 함수 생성
# CrossEntropyLoss는 분류 문제에서 예측값과 실제 정답의 차이를 계산합니다.
loss_fn = nn.CrossEntropyLoss()

# SGD와 AdamW 두 가지 Optimizer를 차례대로 비교합니다.
for name in ["SGD", "AdamW"]:
    # 후보마다 같은 초기 가중치에서 다시 시작합니다. 앞 모델을 이어 학습하지 않습니다.
    # 동일한 초기 조건에서 비교하기 위해 난수 시드를 고정
    torch.manual_seed(42)

    # 신경망 모델 생성
    # nn.Linear(2, 8)
    # 입력 데이터 2개를 받아서 뉴런 8개로 전달
    #
    # nn.ReLU()
    # 활성화 함수로, 음수 값을 0으로 변환
    #
    # nn.Linear(8, 2)
    # 뉴런 8개의 결과를 받아 최종적으로 2개 클래스를 예측
    #
    # [Weight와 Bias]
    # 각 nn.Linear에는 가중치(Weight)와 편향(Bias)이 존재
    #
    # Weight(가중치):
    # 입력값에 곱해지는 값으로, 각 입력이 결과에 미치는 영향력을 조절
    #
    # Bias(편향):
    # 가중치를 곱해 계산한 결과에 추가로 더해지는 값
    # 양수 또는 음수가 될 수 있으며 결과를 조정하는 역할
    #
    # 가중치와 편향은 처음부터 정답으로 정해져 있는 것이 아니라
    # 모델이 학습하면서 점차 갱신하는 대상
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    # Optimizer 선택
    # Optimizer란 학습 과정에서 가중치와 편향을 어떤 규칙으로 갱신할지 결정하는 최적화 알고리즘
    if name == "SGD":
        # SGD의 학습할 파라미터는 model.parameters()입니다. lr=0.01, momentum=0.0으로 만드세요.
        # 학습률은 기본값에 기대지 않고 명시하며, 두 Optimizer 모두 weight_decay=0.0으로 비교합니다.
        # >>> B01
        # TODO B01: 위 요구사항에 맞게 이 블록만 완성합니다.
        # SGD Optimizer
        # model.parameters(): 모델 내부의 학습 대상인 가중치와 편향
        # lr=0.01: 학습률을 0.01로 설정
        # momentum=0.0: 이전 기울기의 영향을 사용하지 않음
        # weight_decay=0.0: 가중치 감쇠를 적용하지 않음
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
        # <<< B01

    else:
        # AdamW를 같은 lr=0.01로 만드세요. weight_decay=0.0을 반드시 적으세요.
        # 이번에는 감쇠 효과가 아닌 갱신 규칙을 비교합니다. 이 공통 lr이 두 알고리즘의 최적 lr이라는 뜻은 아닙니다.
        # >>> B02
        # TODO B02: 위 요구사항에 맞게 이 블록만 완성합니다.
        # AdamW Optimizer
        # SGD와 다른 방식으로 기울기를 활용하여 가중치와 편향을 갱신
        # lr=0.01: 학습률을 0.01로 설정
        # weight_decay=0.0: 가중치 감쇠를 적용하지 않음
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        # <<< B02

    # 총 20번 반복하여 모델 학습
    # 한 번의 반복을 Epoch라고 함  # 이전 기울기 정리 > 예측 > 오차계산 > 기울기계산 > 가중치 갱신
    for epoch in range(20):     # zero_grad() > forward > loss > backward > step
        model.train()           # 학습 모드로 전환
        optimizer.zero_grad()   # 이전에 계산된 기울기를 초기화, 이전 기울기 정리

        # 예측
        # 학습 데이터를 모델에 입력하여 예측값 계산
        # 실제 정답(y_train)과 비교하여 손실(Loss) 계산
        loss = loss_fn(model(X_train), y_train)

        # 오차계산
        # 손실을 기준으로 기울기(Gradient) 계산
        # 가중치와 편향을 어느 방향으로 얼마나 수정해야 할지 계산하는 과정
        loss.backward()

        # 가중치 갱신
        # Optimizer가 계산된 기울기를 이용하여
        # 모델의 가중치와 편향을 실제로 갱신
        optimizer.step()
    model.eval()    # 학습 종료 후 평가 모드로 전환

    # 검증 단계에서는 기울기를 계산하지 않음
    with torch.no_grad():
        # 검증 데이터를 이용하여 손실 계산
        # item()을 사용하여 Tensor 값을 일반 숫자로 변환
        valid_loss = loss_fn(model(X_valid), y_valid).item()

    # Optimizer별 검증 손실 출력
    print(name, "20 epochs / valid loss:", round(valid_loss, 6))
