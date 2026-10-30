from pathlib import Path
import pandas as pd
import torch
from torch import nn

# PyTorch가 CPU에서 사용할 스레드 수를 1개로 제한
# 실행 환경에 따라 연산 결과와 실행 속도를 일정하게 관리하기 위한 설정
torch.set_num_threads(1)
base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 그래프 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu")
df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

# 학습, 검증 데이터의 입력값 추출
# x1, x2 두 개의 특성을 가져옴
# PyTorch에서 사용할 수 있도록 Float32 텐서로 변환
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)

# 학습, 검증 데이터의 정답값 추출
# label은 분류할 클래스 번호
# CrossEntropyLoss에서 사용할 수 있도록 정수형(Long)으로 변환
y_train = torch.tensor(train["label"].values, dtype=torch.long)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)

# 분류 문제에서 사용하는 손실 함수
# 모델의 예측값과 실제 정답 사이의 차이를 계산
loss_fn = nn.CrossEntropyLoss()

# AdamW의 lr 후보 0.001, 0.01, 0.1을 세 설정으로 만드세요.
# 실행 순서·seed·모델 폭·데이터는 바꾸지 않습니다.
# >>> A02
# TODO A02: 위 요구사항에 맞게 이 블록만 완성합니다.
# 각 설정의 이름과 학습률을 딕셔너리로 저장
configs = [{'name': 'lr0.001', 'lr': 0.001}, {'name': 'lr0.01', 'lr': 0.01},
            {'name': 'lr0.1', 'lr': 0.1}]
# <<< A02
summary = [] # 각 학습률별 실험 결과를 저장할 빈 리스트

# configs에 저장된 설정을 하나씩 꺼내서 반복(45~46라인 참고)
for config in configs:
    # 매 실험마다 동일한 난수 시드 설정
    # 모델의 초기 가중치와 편향을 동일하게 만들어 공정하게 비교
    torch.manual_seed(42)

    # 모델의 은닉층 뉴런 수 설정
    # width가 설정에 없으면 기본값인 32 사용
    width = config.get("width", 32)

    # 첫 번째 입력층 생성
    # 입력 특성 2개(x1, x2)를 받아 width개의 출력 생성
    layers = [nn.Linear(2, width)]

    # 설정에 bn이 True라면 Batch Normalization 추가
    # 현재 configs에는 bn이 없으므로 기본값 False가 적용되어 추가되지 않음
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    # 은닉층에 ReLU와 Dropout 추가
    layers.extend([ 
        nn.ReLU(), # ReLU: 음수 값을 0으로 바꾸는 활성화 함수
        # Dropout: 학습 중 일부 뉴런을 무작위로 비활성화하여 과적합을 줄이는 방법
        nn.Dropout(config.get("dropout", 0.0)), 
        nn.Linear(width, width) # 현재 dropout 설정이 없으므로 기본값 0.0 사용
    ])

    # 두 번째 Batch Normalization 추가 여부 확인
    # 현재 설정에서는 추가되지 않음
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # 두 번째 은닉층과 마지막 출력층 추가
    # 마지막 nn.Linear(width, 2)는 2개 클래스에 대한 예측값을 출력
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    
    # layers에 저장된 층들을 하나의 신경망 모델로 연결
    # 모델을 지정한 장치(CPU)로 이동
    model = nn.Sequential(*layers).to(device)

    # 학습 데이터를 묶어서 배치 단위로 제공하는 DataLoader 생성
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    # DataLoader()는 학습 데이터를 모델에 배치(Batch) 단위로 나눠서 전달해 주는 함수
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,  # 한 번에 데이터 32개씩 학습
        shuffle=True,   # 학습할 때마다 데이터 순서를 섞음

        # 후보마다 동일한 seed를 사용하여 데이터 섞는 순서도 동일하게 유지
        generator=torch.Generator().manual_seed(43),

        num_workers=0  # 데이터를 불러오는 별도 프로세스를 사용하지 않음
    )

    # optimizer 설정이 SGD라면 SGD 사용
    # 현재 configs에는 optimizer가 없으므로 기본값 AdamW 사용
    if config.get("optimizer", "AdamW") == "SGD":

        optimizer = torch.optim.SGD(
            model.parameters(),  # 모델의 학습 가능한 가중치와 편향
            lr=config["lr"],     # 현재 실험의 학습률
            momentum=config.get("momentum", 0.0),
            weight_decay=0.0
        )
    else:
        # 현재 실험에서 사용하는 AdamW 옵티마이저
        optimizer = torch.optim.AdamW(
            model.parameters(),

            # configs에서 현재 반복 중인 학습률을 가져옴
            lr=config["lr"],

            # 가중치 감쇠를 사용하지 않도록 0으로 설정
            weight_decay=0.0
        )

    history = [] # Epoch별 학습 결과를 저장할 빈 리스트

    
    # 총 20번 반복하여 모델 학습
    # Epoch는 전체 학습 데이터를 한 번 학습한 단위
    for epoch in range(1, 20 + 1):

        # 모델을 학습 모드로 전환
        # Dropout이나 BatchNorm이 학습 방식으로 동작
        model.train()

        # DataLoader에서 배치 단위로 데이터를 가져옴
        for xb, yb in loader:

            # 입력 데이터와 정답 데이터를 CPU 또는 GPU로 이동
            xb, yb = xb.to(device), yb.to(device)

            # 이전 학습에서 계산된 기울기를 초기화
            # 이전 기울기가 누적되지 않도록 함
            optimizer.zero_grad()

            # 모델에 입력 데이터를 전달하여 예측값 계산
            # 예측값과 실제 정답을 비교하여 손실 계산
            loss = loss_fn(model(xb), yb)

            # 손실을 바탕으로 가중치와 편향의 기울기 계산
            loss.backward()

            # 계산된 기울기를 바탕으로 가중치와 편향 업데이트
            optimizer.step()

        # 한 Epoch의 학습이 끝나면 평가 모드로 전환
        # Dropout 비활성화, BatchNorm 평가 방식 적용
        model.eval()

        # 평가 과정에서는 기울기를 계산하지 않음
        # 불필요한 연산과 메모리 사용을 줄임
        with torch.no_grad():

            # 학습 데이터를 다시 모델에 넣어 손실 계산
            # eval 모드에서 측정한 학습 손실
            train_loss = loss_fn(
                model(X_train.to(device)),
                y_train.to(device)
            ).item()

            # 검증 데이터로 예측 수행
            logits = model(X_valid)

            # 검증 데이터의 손실 계산
            valid_loss = loss_fn(logits, y_valid).item()

            # 예측 결과에서 가장 높은 점수를 가진 클래스 선택
            # 실제 정답과 비교하여 정확도 계산
            # mean()으로 전체 데이터 중 정답을 맞힌 비율 계산
            valid_acc = (
                logits.argmax(1) == y_valid
            ).float().mean().item()

        # 현재 Epoch의 학습 손실, 검증 손실, 정확도를 history에 저장
        history.append([
            epoch,
            train_loss,
            valid_loss,
            valid_acc
        ])

    # 1. 학습 결과 정리
    # history에 저장된 Epoch별 학습 결과를 DataFrame으로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()] # 검증 손실이 가장 낮았던 Epoch의 결과를 best에 저장
    
    # 2. 실험 결과 저장
    name = config["name"]# 현재 실험의 이름을 가져옴
    table.to_csv(out / f"02_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})

    # 3. 학습 결과 그래프 생성 및 저장
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"02_{name}_curves.png", dpi=150)
    plt.close()
    
    # 4. 실험 결과 출력 및 최종 정리
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
result = pd.DataFrame(summary)
result.to_csv(out / "02_comparison.csv", index=False)
print("생성: outputs/02_*.csv, *.png / test 미사용")
