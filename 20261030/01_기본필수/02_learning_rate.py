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

# 학습률 후보 두 개를 리스트로 쓰세요: 0.001, 0.01.
# 모델·seed·20 epochs·AdamW·weight_decay=0은 그대로 둡니다.
# >>> B03
# TODO B03: 위 요구사항에 맞게 이 블록만 완성합니다.
learning_rates = [0.001, 0.01]
# <<< B03
loss_fn = nn.CrossEntropyLoss() # 분류 문제에서 예측값과 실제 정답의 차이를 계산하는 손실 함수

# 리스트에 저장된 학습률을 하나씩 꺼내 실험
for lr in learning_rates:
    # 동일한 초기 가중치와 편향에서 시작하도록 난수 시드 고정
    torch.manual_seed(42)

    # 신경망 모델 생성
    model = nn.Sequential(
        nn.Linear(2, 8),    # 입력층: 입력 데이터 2개, 은닉층: 뉴런 8개
        nn.ReLU(),          # ReLU: 활성화 함수
        nn.Linear(8, 2)     # 출력층: 2개 클래스 예측
    )


    # AdamW Optimizer 생성
    optimizer = torch.optim.AdamW(
        model.parameters(), # model.parameters(): 학습 대상인 가중치와 편향
        lr=lr,              # lr=lr: 현재 반복에서 선택한 학습률 적용
        weight_decay=0.0    # weight_decay=0.0: 가중치 감쇠 미적용
    )
    losses = []# Epoch별 검증 손실을 기록할 빈 리스트

    # 총 20회 반복하여 모델 학습
    for epoch in range(20):
        model.train() # 학습 모드 전환

        optimizer.zero_grad() # 이전 학습에서 계산된 기울기 초기화

        # 학습 데이터를 모델에 입력하여 예측값 계산
        # 실제 정답과 비교하여 손실 계산
        # backward()를 통해 가중치와 편향의 기울기 계산
        loss_fn(model(X_train), y_train).backward()

        # Optimizer를 사용하여 가중치와 편향 갱신
        optimizer.step()

        # 갱신이 끝난 모델의 valid loss를 기록합니다. valid로 backward하지 않습니다.
        model.eval() # 평가 모드로 전환

        # 검증 단계에서는 기울기를 계산하지 않음
        with torch.no_grad():
            # 검증 데이터의 손실을 계산
            # item()으로 Tensor 값을 일반 숫자로 변환
            # losses 리스트에 현재 Epoch의 검증 손실 추가
            losses.append(loss_fn(model(X_valid), y_valid).item())

    print("lr:", lr, "처음 5개:", [round(x, 6) for x in losses[:5]])
    print("마지막 valid loss:", round(losses[-1], 6))
