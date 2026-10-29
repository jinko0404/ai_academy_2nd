"""실습 06: 80 epoch 학습하고 10 epoch마다 같은 시점의 지표를 기록합니다."""
"""
[실습 목적]
- CSV 파일에서 학습 데이터와 검증 데이터를 불러온다.
- PyTorch로 인공신경망(ANN) 모델을 생성한다.
- 학습 데이터를 이용해 모델을 80 epoch 동안 학습한다.
- 10 epoch마다 학습 손실, 검증 손실, 정확도를 측정한다.
- 측정 결과를 CSV 파일로 저장한다.

[전체 실행 순서]
1. 필요한 라이브러리 불러오기
2. CSV 파일에서 데이터 읽기
3. 학습 데이터와 검증 데이터 분리
4. 학습 환경 및 모델 설정
5. 모델 학습 및 성능 측정
6. 학습 결과를 CSV 파일로 저장

[핵심 용어]
- epoch: 전체 학습 데이터를 한 번 학습하는 단위
- batch: 데이터를 나누어 학습할 때 사용하는 작은 묶음
- ANN: 인공신경망(Artificial Neural Network)
- loss: 모델의 예측과 실제 정답 사이의 차이를 나타내는 값
- 역전파: 손실을 기준으로 모델의 기울기를 계산하는 과정
- 순전파: 입력 데이터를 모델에 넣어서 예측 결과를 계산하는 과정
- optimizer: 계산된 기울기를 이용해 모델의 가중치를 갱신하는 도구
- train_loss: 학습 데이터에서 계산한 손실
- valid_loss: 검증 데이터에서 계산한 손실
- accuracy: 검증 데이터에서 정답을 맞힌 비율
- SGD 최적화 : 모델의 가중치를 조금씩 수정하면서 학습시키는 최적화 방법

[주의 사항]
- test 데이터 40건은 모델 학습과 검증에 사용하지 않는다.
- 검증 데이터는 성능 확인에만 사용하며 가중치 갱신에는 사용하지 않는다.
- epoch 0은 학습 전 성능을 측정한다.
"""


from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

base = Path(__file__).resolve().parent
out = base / "outputs"
out.mkdir(exist_ok=True)
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 학습 데이터를 입력과 정답 한 쌍으로 묶는다.
# batch_size=32는 한 번에 최대 32개 데이터로 학습한다는 뜻이다.
# shuffle=True는 학습할 때 데이터 순서를 섞는다는 뜻이다.
# generator의 시드를 43으로 설정해 섞는 순서를 재현하기 쉽게 한다.
# num_workers=0은 별도의 데이터 로딩 작업 프로세스를 사용하지 않는다.
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)

# 인공신경망(ANN) 모델을 만든다.
# nn.Linear(2, 8): 입력 특성 2개를 받아 8개 값으로 변환한다.
# nn.ReLU(): 음수 값을 0으로 바꾸는 활성화 함수다.
# nn.Linear(8, 2): 8개 값을 두 개의 클래스 점수로 변환한다.
# 마지막 .to(device)는 모델을 GPU 또는 CPU로 이동한다.
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)

# 분류 문제에 사용할 손실 함수다.
# 모델의 클래스별 예측 점수와 실제 정답을 비교해 손실을 계산한다.
loss_fn = nn.CrossEntropyLoss()

# SGD 최적화 도구를 만든다.
# SGD는 모델의 가중치를 조금씩 수정하면서 학습시키는 최적화 방법
# model.parameters()는 모델의 학습 가능한 가중치와 편향을 전달한다.
# lr=0.1은 학습률로, 가중치를 얼마나 크게 갱신할지 결정한다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 학습 데이터를 평가에 사용할 수 있도록 지정한 장치로 이동한다.
Xt, yt = X_train.to(device), y_train.to(device)

# 검증 데이터도 모델과 같은 장치로 이동한다.
Xv, yv = X_valid.to(device), y_valid.to(device)


# epoch마다 측정한 결과를 저장할 빈 리스트를 만든다.
# 각 항목에는 [epoch, train_loss, valid_loss, accuracy]가 들어간다.
history = []


# epoch를 0부터 80까지 총 81번 반복한다.
# epoch 0은 학습 전 성능 측정용이고, 1~80에서 실제 학습한다.
for epoch in range(81):

    # epoch가 0보다 클 때만 모델을 학습한다.
    if epoch > 0:

        # 모델을 학습 모드로 설정한다.
        model.train()

        # 학습 데이터를 batch 단위로 가져온다.
        for xb, yb in loader:

            # 현재 batch의 입력과 정답을 GPU 또는 CPU로 이동한다.
            xb, yb = xb.to(device), yb.to(device)

            # 이전 batch에서 계산된 기울기를 초기화한다.
            optimizer.zero_grad()

            # 순전파를 수행해 예측 점수를 계산하고 손실을 구한다.
            loss = loss_fn(model(xb), yb)

            # 역전파를 수행해 손실에 대한 모델 파라미터의 기울기를 계산한다.
            loss.backward()

            # 계산된 기울기를 이용해 모델의 가중치와 편향을 갱신한다.
            optimizer.step()

    # epoch가 0, 10, 20, ... , 80일 때 성능을 측정한다.
    if epoch % 10 == 0:

        # 모델을 평가 모드로 설정한다.
        model.eval()

        # 평가 단계에서는 기울기를 계산하지 않는다.
        # 불필요한 계산을 줄이고 메모리를 절약한다.
        with torch.no_grad():

            # 학습 데이터에 대한 손실을 계산한다.
            # model(Xt)는 학습 데이터의 예측 점수를 계산한다.
            # loss_fn은 예측 점수와 실제 정답 yt를 비교한다.
            # item()은 Tensor에서 숫자 하나를 Python 숫자로 가져온다.
            train_loss = loss_fn(model(Xt), yt).item()

            # 검증 데이터를 모델에 넣어 클래스별 예측 점수(Logits)를 계산한다.
            logits = model(Xv)

            # 검증 데이터의 예측 점수와 실제 정답을 비교해 손실을 계산한다.
            valid_loss = loss_fn(logits, yv).item()

            # 각 데이터에서 점수가 가장 높은 클래스 번호를 선택한다.
            # dim=1은 각 데이터 행에서 가장 높은 점수를 찾는다는 뜻이다.
            # 실제 정답 yv와 비교해 예측이 맞았는지 확인한다.
            # True는 1, False는 0으로 변환하고 평균을 계산해 정확도를 구한다.
            accuracy = (
                (logits.argmax(dim=1) == yv)
                .float()
                .mean()
                .item()
            )

        # 현재 epoch와 학습 손실, 검증 손실, 정확도를 history에 저장한다.
        history.append([
            epoch,
            train_loss,
            valid_loss,
            accuracy
        ])

        # 측정 결과를 화면에 출력한다.
        # 손실은 소수점 아래 6자리, 정확도는 소수점 아래 4자리까지 표시한다.
        print(
            epoch,
            round(train_loss, 6),
            round(valid_loss, 6),
            round(accuracy, 4)
        )


with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"])
    writer.writerows(history)
