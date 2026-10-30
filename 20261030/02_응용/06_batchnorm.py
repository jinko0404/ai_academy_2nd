from pathlib import Path
import pandas as pd
import torch
from torch import nn

# ==================================================
# 1. 기본 환경 설정
# ==================================================

torch.set_num_threads(1) # CPU 연산 스레드 수 제한
base = Path(__file__).resolve().parent
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.
device = torch.device("cpu") # CPU 사용

# CSV 파일을 읽고 학습 데이터와 검증 데이터 분리
df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

# 입력 데이터(X)와 정답 데이터(y)를 Tensor로 변환
# float32: 실수형 데이터 / long: 정수형 데이터
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)

loss_fn = nn.CrossEntropyLoss()  # 분류 모델의 손실 함수

# ==================================================
# 2. 모델별 실험 설정
# ==================================================
# bn=False와 bn=True만 다른 두 설정을 구성하세요.
# batch_size=32는 동일하고 마지막 16건도 학습합니다. BatchNorm을 사용하는 train batch는 2건 이상이어야 합니다.
# >>> A06
# TODO A06: 위 요구사항에 맞게 이 블록만 완성합니다.
# baseline: BatchNorm 사용 안 함
# batchnorm: BatchNorm 사용
#
# batch_size=32로 설정하고 마지막 데이터도 학습
# BatchNorm을 사용하는 경우 Batch 데이터가 최소 2개 이상이어야 함
configs = [{'name': 'baseline', 'lr': 0.01, 'bn': False},
           {'name': 'batchnorm', 'lr': 0.01, 'bn': True}]
# <<< A06
summary = [] # 각 모델의 실험 결과를 저장할 리스트

# configs에 설정된 모델을 하나씩 가져와 학습
for config in configs:
    
    torch.manual_seed(42)  # 모델 초기 가중치의 난수 시드 고정

    width = config.get("width", 32)  # 은닉층 뉴런 수, 기본값은 32

    # [모델 구조]
    # 입력층 → 은닉층 → 은닉층 → 출력층
    # 입력 데이터 2개를 받아 최종적으로 2개 클래스를 예측
    layers = [nn.Linear(2, width)]

    # BatchNorm 설정이 True인 경우에만 정규화 층 추가
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # ReLU: 활성화 함수
    # Dropout: 학습 중 일부 뉴런의 출력을 무작위로 비활성화
    # Linear: 다음 은닉층으로 데이터 전달
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])

    # 두 번째 은닉층에도 BatchNorm 적용
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # 활성화 함수와 Dropout 적용 후 최종 출력층 연결
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])

    # 정의한 층들을 순서대로 연결하여 모델 생성
    model = nn.Sequential(*layers).to(device)

    # ==================================================
    # 4. 데이터 배치 및 최적화 알고리즘 설정
    # ==================================================

    # DataLoader: 전체 데이터를 Batch 단위로 나누어 공급
    # batch_size=32: 한 번에 최대 32개 데이터 사용
    # shuffle=True: 학습 데이터를 무작위 순서로 섞음
    # generator: 데이터 섞기에 사용하는 난수 시드 고정
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)

    # Optimizer: 모델의 가중치를 업데이트하는 방법
    # SGD 설정이 있으면 SGD 사용, 없으면 AdamW 사용
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)

    history = [] # Epoch별 학습 결과 저장

    # ==================================================
    # 5. 모델 학습 및 검증
    # ==================================================

    # 총 40 Epoch 동안 학습 반복
    for epoch in range(1, 40 + 1):

        # [Train Mode]
        # 학습 모드로 전환
        # BatchNorm은 현재 Batch의 평균과 분산을 사용
        # Running Mean과 Running Var도 학습 중 갱신
        model.train()

        # DataLoader에서 Batch 단위로 데이터 가져오기
        for xb, yb in loader:

            xb, yb = xb.to(device), yb.to(device)  # 데이터를 실행 장치로 이동

            optimizer.zero_grad()  # 이전 반복에서 계산된 기울기 초기화

            loss = loss_fn(model(xb), yb)  # 모델 예측값과 실제 정답으로 손실 계산

            loss.backward()  # 손실을 바탕으로 기울기 계산

            optimizer.step()  # 계산된 기울기를 이용해 가중치 업데이트


        # [Evaluation Mode]
        # 학습 모드와 평가 모드의 결과를 구분하기 위해 eval 모드로 전환
        # BatchNorm은 학습 중 누적한 Running Mean과 Running Var를 사용
        # Dropout은 평가 모드에서 비활성화
        model.eval()

        # 평가 단계에서는 기울기를 계산하지 않음
        with torch.no_grad():

            # 학습 데이터의 손실 계산
            # eval 모드이므로 Dropout이 비활성화된 상태에서 평가
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()

            # 검증 데이터에 대한 모델 예측 결과
            logits = model(X_valid)

            # 검증 데이터의 손실 계산
            valid_loss = loss_fn(logits, y_valid).item()

            # 예측값 중 가장 높은 값의 클래스를 선택
            # 실제 정답과 비교하여 평균 정확도 계산
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()

        # 현재 Epoch의 학습 및 검증 결과 저장
        history.append([epoch, train_loss, valid_loss, valid_acc])

    # Epoch별 결과를 DataFrame으로 변환
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    
    best = table.loc[table["valid_loss"].idxmin()] # Valid Loss가 가장 낮은 Epoch의 결과 선택

    name = config["name"] # 현재 모델 이름 가져오기

    # Epoch별 학습 결과를 CSV 파일로 저장
    table.to_csv(out / f"06_{name}_history.csv", index=False)

    # 모델별 최적 성능 및 마지막 Epoch 결과 저장
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    
    # [Loss 그래프]
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"06_{name}_curves.png", dpi=150)
    plt.close()

    # 모델별 최적 Epoch와 검증 손실 출력
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

# ==================================================
# 7. 두 모델의 최종 비교 결과 저장
# ==================================================

result = pd.DataFrame(summary)
result.to_csv(out / "06_comparison.csv", index=False)
print("생성: outputs/06_*.csv, *.png / test 미사용")

# ==================================================
# 8. BatchNorm Train/Eval 모드 차이 진단
# ==================================================

# 마지막 BN 모델을 복사해 진단합니다. 원래 학습 결과/저장 결과는 바꾸지 않습니다.
# 마지막 BN 모델을 복사하여 별도로 진단
# 원본 모델의 학습 결과와 저장 결과에는 영향을 주지 않음
probe = copy.deepcopy(model)

xb = X_train[:32].to(device)  # 학습 데이터 중 앞에서 32개 추출


# [Train Mode]
# 현재 Batch의 평균과 분산을 사용하여 정규화
# 학습 모드에서의 출력 결과 저장
probe.train()

with torch.no_grad():
    train_output = probe(xb)


# [Eval Mode]
# 학습 중 누적한 Running Mean과 Running Var를 사용하여 정규화
# 평가 모드에서의 출력 결과 저장
probe.eval()

with torch.no_grad():
    eval_output = probe(xb)


# 두 모드의 출력값 차이를 계산
# abs(): 차이를 절댓값으로 변환
# max(): 차이 중 가장 큰 값 선택
# item(): Tensor 값을 일반 숫자로 변환
print("진단용 복사 모델 train/eval 최대 차이:", round((train_output - eval_output).abs().max().item(), 6))
