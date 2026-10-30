from pathlib import Path
import pandas as pd
import torch
from torch import nn

# ==================================================
# 1. 기본 환경 설정
# ==================================================

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

# ==================================================
# 2. 데이터 불러오기 및 분리
# ==================================================

df = pd.read_csv(base / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# ==================================================
# 3. 실험 조건 및 기본 설정
# ==================================================

# 실습 07~08: 기준 lr과 비교 lr만 다릅니다. 더 나쁘게 나와도 그대로 보고합니다.
loss_fn = nn.CrossEntropyLoss()
summary = []

# ==================================================
# 4. 모델 생성 및 학습
# ==================================================

# 두 가지 학습률 설정을 하나씩 가져와 실험
for name, lr in [("baseline", 0.001), ("candidate", 0.01)]:
    torch.manual_seed(42)  # 모델 초기 가중치의 난수 시드 고정

    # [모델 구조]
    # 입력층 → 은닉층 → 출력층
    # 입력 특성 2개를 받아 은닉층 뉴런 8개를 거쳐 2개 클래스를 예측
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    # AdamW: 모델의 가중치를 업데이트하는 최적화 알고리즘
    # lr: 학습률 / weight_decay: 가중치 감소 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0
    )

    history = [] # Epoch별 학습 및 검증 결과 저장

    # ==================================================
    # 5. Epoch 반복 학습
    # ==================================================
    for epoch in range(1, 21): # 총 20회 학습 반복
        model.train()  # 모델을 학습 모드로 전환

        optimizer.zero_grad()  # 이전 학습에서 계산된 기울기 초기화

        # 학습 데이터로 예측하고 실제 정답과 비교하여 손실 계산
        loss = loss_fn(model(X_train), y_train)

        # 이미 zero_grad와 loss 계산이 끝났습니다. loss.backward() 다음 optimizer.step()을 호출하세요.
        # 학습 데이터만 사용합니다. 비교 후보마다 모델을 처음부터 새로 만듭니다.
        # >>> B08
        # TODO B08: 위 요구사항에 맞게 이 블록만 완성합니다.
        loss.backward()     #손실을 바탕으로 기울기 계산
        optimizer.step()    # 계산된 기울기를 이용해 가중치 업데이트
        # <<< B08

        # [평가 단계]
        # 학습이 끝난 모델을 평가 모드로 전환
        # 평가 단계에서는 가중치를 업데이트하지 않음
        model.eval()

        # 평가 중에는 기울기를 계산하지 않음
        with torch.no_grad():

            # 학습 데이터에 대한 손실 계산
            # 학습에 사용한 데이터의 예측 손실 확인
            train_loss = loss_fn(model(X_train), y_train).item()

            # 검증 데이터에 대한 손실 계산
            # 학습에 사용하지 않은 검증 데이터로 성능 확인
            valid_loss = loss_fn(model(X_valid), y_valid).item()

        # 한 epoch의 번호와 갱신 후 train_loss, valid_loss를 기록하세요.
        # history.append([epoch, train_loss, valid_loss]) 형태로 추가합니다.
        # >>> B09
        # TODO B09: 위 요구사항에 맞게 이 블록만 완성합니다.
        history.append([epoch, train_loss, valid_loss])
        # <<< B09

    # ==================================================
    # 6. 학습 결과 정리
    # ==================================================
    
    # Epoch별 결과를 DataFrame 형태로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"])

    # Epoch별 학습 결과를 CSV 파일로 저장
    table.to_csv(base / f"history_{name}.csv", index=False)

    best = table.loc[table["valid_loss"].idxmin()] # Valid Loss가 가장 낮은 Epoch의 결과 선택

    # 모델 이름, 학습률, 최적 Epoch, 최저 Valid Loss 저장
    summary.append([name, lr, int(best["epoch"]), best["valid_loss"]])

    # ==================================================
    # 7. 학습 결과 그래프 생성
    # ==================================================
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base / f"curves_{name}.png", dpi=150)
    plt.close()
    
# ==================================================
# 8. 두 모델 비교 결과 저장 및 출력
# ==================================================
result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"])
result.to_csv(base / "experiment_table.csv", index=False)
print(result.round(6).to_string(index=False))
print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")
