from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1) # PyTorch가 CPU에서 사용할 스레드 수를 1개로 제한
base = Path(__file__).resolve().parent # 현재 파이썬 파일이 있는 폴더의 경로를 가져옴
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

# 분류 문제에서 예측값과 실제 정답의 차이를 계산하는 손실 함수
loss_fn = nn.CrossEntropyLoss()

# 두 설정은 momentum만 0.0과 0.9로 다르게 만드세요.
# optimizer=SGD, lr=0.05, 나머지 설정은 동일하게 유지합니다.
# >>> A01
# TODO A01: 위 요구사항에 맞게 이 블록만 완성합니다.
configs = [{'name': 'momentum0', 'optimizer': 'SGD', 'lr': 0.05, 'momentum': 0.1}, 
           {'name': 'momentum9', 'optimizer': 'SGD', 'lr': 0.05, 'momentum': 0.9}]
# <<< A01
summary = [] # 각 실험의 최종 결과를 저장할 빈 리스트

# configs에 저장된 설정을 하나씩 꺼내 실험
for config in configs:
    # 두 실험이 동일한 초기 가중치와 편향에서 시작하도록 시드 고정
    torch.manual_seed(42)

    # 모델의 은닉층 뉴런 수 설정
    # width가 설정되어 있지 않으면 기본값 32 사용
    width = config.get("width", 32)

    # 첫 번째 신경망 층 생성
    # 입력 특성 2개를 받아 width개의 뉴런으로 전달
    layers = [nn.Linear(2, width)]

    # BatchNorm 사용 여부 확인
    # True라면 배치 정규화 층 추가
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # ReLU 활성화 함수와 Dropout 층 추가
    # Dropout은 학습 중 일부 뉴런을 무작위로 비활성화하여 과적합을 줄이는 방법
    # dropout 값이 없으면 기본값 0.0 사용
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    
    # 두 번째 BatchNorm 층 추가 여부 확인
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # 두 번째 활성화 함수, Dropout, 최종 출력층 추가
    # 최종 출력층은 2개 클래스를 예측
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])

    # layers에 저장된 층들을 순서대로 연결하여 신경망 모델 생성
    # 생성한 모델을 CPU 또는 GPU 장치로 이동
    model = nn.Sequential(*layers).to(device)
    
    # 학습 데이터를 Dataset 형태로 묶고 DataLoader로 전달
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,  # 한 번에 32개 데이터씩 학습
        shuffle=True,   # 학습할 때 데이터 순서를 무작위로 섞음
        generator=torch.Generator().manual_seed(43),  # 데이터 순서 재현을 위한 시드
        num_workers=0   # 데이터 로딩에 별도 프로세스를 사용하지 않음
    )

    if config.get("optimizer", "AdamW") == "SGD": # 설정된 Optimizer 종류 확인
        # SGD Optimizer 생성
        # model.parameters(): 학습 대상인 가중치와 편향
        # lr: 학습률
        # momentum: 이전 기울기의 방향을 반영하는 정도
        # weight_decay=0.0: 가중치 감쇠 미적용
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0.0),
            weight_decay=0.0
        )

    else:
        # SGD가 아닌 경우 AdamW Optimizer 사용
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=0.0
        )

    history = [] # Epoch별 학습 결과를 기록할 빈 리스트
    
    for epoch in range(1, 20 + 1): # 총 20번 반복하여 모델 학습, epoch는 1부터 20까지 진행
        model.train() # 모델을 학습 모드로 전환

        # DataLoader에서 Batch 단위로 학습 데이터를 가져옴
        for xb, yb in loader:
             # 가져온 Batch 데이터를 CPU 또는 GPU 장치로 이동
            xb, yb = xb.to(device), yb.to(device)

            # 이전 반복에서 계산된 기울기 초기화
            optimizer.zero_grad()

            # 모델에 입력 데이터를 전달하여 예측값 계산
            # 실제 정답과 비교하여 손실 계산
            loss = loss_fn(model(xb), yb)

            # 손실을 기준으로 가중치와 편향의 기울기 계산
            loss.backward()

            # Optimizer를 사용하여 가중치와 편향 갱신
            optimizer.step()

        # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
        model.eval() # 한 Epoch의 학습이 끝났으므로 평가 모드로 전환

        # 평가 단계에서는 기울기를 계산하지 않음
        with torch.no_grad():
             # 학습 데이터에 대한 손실 계산
            # 평가 모드에서 측정하므로 train_loss_eval이라고 기록
            train_loss = loss_fn(
                model(X_train.to(device)),
                y_train.to(device)
            ).item()

            # 검증 데이터에 대한 예측값 계산
            logits = model(X_valid)

            # 검증 데이터의 손실 계산
            valid_loss = loss_fn(logits, y_valid).item()

            # 예측값에서 가장 높은 점수를 가진 클래스 선택
            # 실제 정답과 비교하여 정확도 계산
            valid_acc = (
                logits.argmax(1) == y_valid
            ).float().mean().item()

        # 현재 Epoch의 학습 손실, 검증 손실, 검증 정확도를 기록
        history.append([epoch, train_loss, valid_loss, valid_acc])

    # 학습 기록을 DataFrame 형태로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    
    best = table.loc[table["valid_loss"].idxmin()] # 검증 손실이 가장 낮았던 Epoch의 데이터 선택 
    name = config["name"] # 현재 실험의 이름 가져오기
    
    # Epoch별 학습 결과를 CSV 파일로 저장
    table.to_csv(out / f"01_{name}_history.csv", index=False)

    # 최종 실험 결과를 summary 리스트에 추가
    summary.append({
        **config,
        "best_epoch": int(best["epoch"]),               # best_epoch: 검증 손실이 가장 낮았던 Epoch
        "best_valid_loss": float(best["valid_loss"]),   # best_valid_loss: 가장 낮았던 검증 손실
        "valid_acc_at_best": float(best["valid_acc"]),  # valid_acc_at_best: 해당 Epoch의 검증 정확도
        "last_train_loss_eval": train_loss,             # last_train_loss_eval: 마지막 Epoch의 학습 손실
        "last_valid_loss": valid_loss                   # last_valid_loss: 마지막 Epoch의 검증 손실
    })

    # 학습 손실과 검증 손실을 비교하는 그래프 생성
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"01_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
result = pd.DataFrame(summary)
result.to_csv(out / "01_comparison.csv", index=False)
print("생성: outputs/01_*.csv, *.png / test 미사용")
