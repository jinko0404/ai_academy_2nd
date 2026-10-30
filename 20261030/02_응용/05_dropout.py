from pathlib import Path
import pandas as pd
import torch
from torch import nn

# PyTorch가 사용할 CPU 스레드 수를 1개로 제한
torch.set_num_threads(1)

base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")
df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)

# 다중 분류 문제에서 사용하는 손실 함수
# 모델의 예측값과 실제 정답의 차이를 계산
loss_fn = nn.CrossEntropyLoss()
# Dropout 확률 p=0.0과 p=0.3을 비교하세요.
# 평가에서는 model.eval()과 no_grad를 모두 사용하고, best epoch는 valid loss로 찾습니다.
# >>> A05
# TODO A05: 위 요구사항에 맞게 이 블록만 완성합니다.
# 비교할 두 가지 실험 설정을 딕셔너리로 구성
configs = [
    {
        'name': 'dropout0.0',  # 첫 번째 실험 이름
        'lr': 0.01,            # 학습률
        'dropout': 0.0         # Dropout을 적용하지 않음
    },
    {
        'name': 'dropout0.3',  # 두 번째 실험 이름
        'lr': 0.01,            # 첫 번째 실험과 동일한 학습률
        'dropout': 0.3         # 학습 중 뉴런의 30%를 무작위로 비활성화
    }
]
# <<< A05
summary = [] # 각 실험의 최종 결과를 저장할 빈 리스트

for config in configs: # configs에 저장된 실험 설정을 하나씩 꺼내서 반복 실행
    torch.manual_seed(42)   # 실험마다 동일한 난수 시드를 사용하여 초기 조건을 맞춤

    width = config.get("width", 32) # 설정에 width가 있으면 해당 값을 사용하고, 없으면 기본값 32 사용

    # 신경망의 첫 번째 층 생성
    # 입력 데이터 2개(x1, x2)를 width 크기의 출력으로 변환
    layers = [nn.Linear(2, width)]

    if config.get("bn", False): # 설정에서 bn이 True라면 Batch Normalization 추가
        layers.append(nn.BatchNorm1d(width))
    
    # ReLU 활성화 함수와 Dropout을 추가하고 다음 Linear 층 연결
    # Dropout 확률은 현재 config에 설정된 값을 사용
    layers.extend([
        nn.ReLU(),
        nn.Dropout(config.get("dropout", 0.0)),
        nn.Linear(width, width)
    ])

    if config.get("bn", False): # 두 번째 층 뒤에도 Batch Normalization 설정이 True라면 추가
        layers.append(nn.BatchNorm1d(width))
    
    # 두 번째 ReLU와 Dropout을 추가한 뒤 최종 출력층 연결
    # 마지막 출력층은 2개의 클래스를 예측하므로 출력 크기가 2
    layers.extend([
        nn.ReLU(),
        nn.Dropout(config.get("dropout", 0.0)),
        nn.Linear(width, 2)
    ])

    # layers에 들어 있는 층들을 하나의 신경망으로 연결
    # 생성한 모델을 지정된 device로 이동
    model = nn.Sequential(*layers).to(device)
    
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    loader = DataLoader(
    TensorDataset(X_train, y_train),  # 학습 데이터와 정답 데이터를 하나의 데이터셋으로 묶음
    batch_size=32,                    # 한 번에 32개씩 데이터를 가져옴
    shuffle=True,                     # 학습 데이터를 무작위 순서로 섞음
    generator=torch.Generator().manual_seed(43),  # 데이터 섞기에 사용할 난수 시드를 43으로 고정
    num_workers=0                     # 데이터를 불러오는 작업에 별도 프로세스를 사용하지 않음
    )
    # 설정된 optimizer가 SGD라면 SGD 옵티마이저 사용
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0.0),
            weight_decay=0.0
        )
    # 별도 설정이 없다면 AdamW 옵티마이저 사용
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.0
        )

    history = [] # 매 epoch마다 측정한 손실과 정확도를 저장할 리스트

    for epoch in range(1, 70 + 1): # 총 40번 반복하여 모델 학습
        # 모델을 학습 모드로 변경
        # Dropout이 설정된 경우 일부 뉴런이 무작위로 비활성화됨
        model.train()

        for xb, yb in loader:   # DataLoader에서 학습 데이터를 batch 단위로 가져옴
            xb, yb = xb.to(device), yb.to(device)   # 입력 데이터와 정답 데이터를 지정된 device로 이동
            optimizer.zero_grad()                   # 이전 학습에서 계산된 기울기를 초기화

            # 모델에 입력 데이터를 전달하여 예측값 생성
            # 예측값과 실제 정답을 비교하여 손실 계산
            loss = loss_fn(model(xb), yb)

            loss.backward()     # 손실을 기준으로 각 파라미터의 기울기 계산
            optimizer.step()    # 계산된 기울기를 이용해 모델의 가중치와 편향 업데이트

        # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
        # 모델을 평가 모드로 변경
        # Dropout이 비활성화되고 BatchNorm도 평가 방식으로 동작
        model.eval()

        # 평가 과정에서 기울기를 계산하지 않도록 설정
        # 메모리 사용량과 불필요한 연산을 줄임
        with torch.no_grad():
            # 학습 데이터를 평가 모드로 전달하여 손실 계산
            # 학습 중 Dropout이 적용된 손실이 아니라 평가 모드의 손실
            train_loss = loss_fn(
                model(X_train.to(device)),
                y_train.to(device)
            ).item()

            # 검증 데이터를 모델에 전달하여 예측 결과 생성
            logits = model(X_valid)

            # 예측 결과와 실제 정답을 비교하여 검증 손실 계산
            valid_loss = loss_fn(logits, y_valid).item()

            # 각 데이터의 가장 높은 예측 점수를 가진 클래스 선택
            # 실제 정답과 비교하여 평균 정확도 계산
            valid_acc = (
                logits.argmax(1) == y_valid
            ).float().mean().item()

        # 현재 epoch 번호와 측정 결과를 history에 저장
        history.append([epoch, train_loss, valid_loss, valid_acc])  


    # history에 저장된 결과를 표 형태의 DataFrame으로 변환
    # 각 열에 epoch, 학습 손실, 검증 손실, 검증 정확도 이름 지정
    table = pd.DataFrame(
        history,
        columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"]
    )

    # 검증 손실(valid_loss)이 가장 낮았던 행을 찾음
    # 해당 epoch를 가장 좋은 학습 시점으로 판단
    best = table.loc[table["valid_loss"].idxmin()]

    name = config["name"]   # 현재 실험의 이름을 config에서 가져옴

    table.to_csv(out / f"05_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"05_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
result = pd.DataFrame(summary)
result.to_csv(out / "05_comparison.csv", index=False)
print("생성: outputs/05_*.csv, *.png / test 미사용")
