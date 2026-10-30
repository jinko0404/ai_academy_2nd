from pathlib import Path
import pandas as pd
import torch
from torch import nn

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

# CSV 데이터 불러오기
df = pd.read_csv(base / "data" / "points.csv")

# 전체 데이터에서 학습용과 검증용 데이터를 분리합니다.
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

# 입력 데이터(X)와 정답 데이터(y)를 PyTorch Tensor로 변환합니다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)

# 분류 문제에서 사용하는 손실 함수입니다.
# CrossEntropyLoss는 모델의 예측값과 실제 정답을 비교합니다.
loss_fn = nn.CrossEntropyLoss()

# 비교할 세 가지 실험 조건
# SGD: SGD Optimizer 사용
# AdamW: AdamW Optimizer 사용
# AdamW + Dropout: AdamW에 Dropout 적용
configs = [{'name': 'sgd', 'optimizer': 'SGD', 'lr': 0.01}, {'name': 'adamw', 'lr': 0.01}, {'name': 'adamw_dropout', 'lr': 0.01, 'dropout': 0.3}]

summary = [] # 각 실험의 최종 결과를 저장할 빈 리스트

for config in configs:
    torch.manual_seed(42)  # 후보마다 같은 난수 시드를 사용해 초기화를 재현합니다.

    width = config.get("width", 32)  # 설정에 width가 없으면 뉴런 32개를 사용합니다.

    # 신경망의 첫 번째 층: 입력 특성 2개를 32개 특성으로 변환합니다.
    layers = [nn.Linear(2, width)]

    # 설정에서 bn이 True라면 BatchNorm을 추가합니다.
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # ReLU는 활성화 함수, Dropout은 일부 뉴런을 무작위로 비활성화합니다.
    # 마지막 Linear는 다음 은닉층으로 전달할 특성을 만듭니다.
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])

    # 두 번째 은닉층 뒤에도 BatchNorm을 적용할 수 있습니다.
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))

    # 마지막 출력층은 클래스 0과 1에 대한 점수를 출력합니다.
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])

    # 구성한 층들을 하나의 신경망으로 연결하고 CPU/GPU 장치로 이동합니다.
    model = nn.Sequential(*layers).to(device)

    # DataLoader는 데이터를 Batch 단위로 나누어 학습에 제공합니다.
    # batch_size=32: 한 번에 데이터 32개씩 학습합니다.
    # shuffle=True: 매 Epoch마다 학습 데이터 순서를 섞습니다.
    # Generator seed를 동일하게 설정해 후보마다 같은 순서로 섞이도록 합니다.
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)

    # 설정에 따라 Optimizer를 선택합니다.
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)
    
    history = [] # Epoch마다 Train Loss, Valid Loss, Accuracy를 기록합니다.
    
    best_loss = float("inf")  # 처음에는 가장 작은 Loss를 무한대로 설정합니다.
    best_state = None  # 가장 성능이 좋았을 때의 모델 가중치를 저장할 변수입니다.
    best_epoch = 0  # 가장 좋은 성능이 나온 Epoch를 저장합니다.

    for epoch in range(1, 40 + 1): # 총 40 Epoch 동안 학습합니다.
        model.train() # 학습 모드: Dropout 활성화, BatchNorm 학습 통계 갱신
        
        for xb, yb in loader: 
            xb, yb = xb.to(device), yb.to(device) # 현재 Batch 데이터를 장치로 이동합니다.
            
            optimizer.zero_grad()  # 이전 단계에서 계산한 기울기를 초기화합니다.
            loss = loss_fn(model(xb), yb)  # 모델의 예측값과 실제 정답으로 Loss를 계산합니다.
            # loss.backward() 다음 optimizer.step()을 호출하세요.
            # 같은 초기화·동일 shuffle 순서를 사용하며 세 후보의 변경 조건을 읽어 구분합니다.
            # >>> A07
            # TODO A07: 위 요구사항에 맞게 이 블록만 완성합니다.
            loss.backward() # Loss를 바탕으로 각 가중치의 기울기를 계산합니다.
            optimizer.step() # 계산된 기울기를 이용해 모델의 가중치를 갱신합니다.
            # <<< A07

        # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
        # epoch 종료 후 model.eval()을 호출하세요.
        # 그 아래 no_grad 블록은 gradient 추적을 끄지만, Dropout 모드를 바꾸지는 않습니다.
        # >>> A08
        # TODO A08: 위 요구사항에 맞게 이 블록만 완성합니다.
        # 한 Epoch의 학습이 끝나면 평가 모드로 전환합니다.
        # Dropout을 비활성화하고 BatchNorm은 저장된 평균·분산을 사용합니다.
        model.eval()
        # <<< A08

        # 평가 과정에서는 기울기를 계산할 필요가 없습니다.
        with torch.no_grad():
            # 학습 데이터에 대한 Loss를 계산
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            
            logits = model(X_valid)# 검증 데이터에 대한 모델 예측값을 계산합니다.

            valid_loss = loss_fn(logits, y_valid).item() # 검증 데이터의 Loss를 계산합니다.

            # 가장 높은 점수를 가진 클래스를 예측 결과로 선택합니다.
            # 실제 정답과 비교하여 정확도(Accuracy)를 계산합니다.
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()

            # 최저 valid loss를 관측하면 값·epoch·가중치 복사본을 함께 보관합니다.
            # A09 요구: 동률은 먼저 발견한 epoch 유지. copy.deepcopy(model.state_dict()) 사용.
            # model.state_dict()만 대입하면 이후 학습이 같은 Tensor를 바꾸므로 안 됩니다.
            # >>> A09
            # TODO A09: 최적 loss·epoch와 독립된 state_dict 복사본을 보관하세요.
            if valid_loss < best_loss:  # 현재 Valid Loss가 기존 최저 Loss보다 낮은지 확인합니다.
                best_loss = valid_loss  # 가장 낮은 Valid Loss 갱신
                best_epoch = epoch      # 최저 Loss가 나온 Epoch 기록
                best_state = copy.deepcopy(model.state_dict())  # 해당 시점의 모델 가중치를 독립적으로 복사하여 저장
            # <<< A09

        # 현재 Epoch의 학습 결과를 리스트에 저장합니다.
        history.append([epoch, train_loss, valid_loss, valid_acc])
    
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"07_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    
    # 제공 코드: 학습이 끝난 뒤 최적 상태로 복원하고 그 모델로 다시 평가합니다.
    model.load_state_dict(best_state)

    model.eval()# 복원한 모델을 평가 모드로 전환합니다.

    with torch.no_grad():
         # 복원한 모델로 검증 데이터의 예측 결과를 계산합니다.
        best_pred = model(X_valid).argmax(1).cpu()

        # 복원된 모델의 Valid Loss를 다시 계산합니다.
        restored_loss = loss_fn(model(X_valid), y_valid).item()

    labels = y_valid.cpu()  # 실제 정답을 CPU로 이동합니다.
    class_rows = []  # 클래스별 Recall 결과를 저장합니다.

    # 클래스 0과 클래스 1을 각각 평가합니다.
    for label in (0, 1):
        # A10 요구: 실제 정답이 label인 행을 먼저 고른 뒤 support·correct·recall 계산.
        # 전체 accuracy나 '예측 label의 수'로 나누지 않습니다. 0건이면 recall=None.
        # >>> A10
        # TODO A10: 클래스별 실제 표본 수·정답 수·Recall을 기록하세요.
        # 실제 정답이 현재 label과 같은 데이터 선택
        mask = labels == label

        # 실제 해당 클래스의 데이터 개수
        support = mask.sum().item()

        # 실제 정답과 예측이 모두 일치하는 데이터 개수
        correct = ((best_pred == label) & mask).sum().item()

        # Recall 계산 (실제 데이터가 0개면 None)
        recall = correct / support if support > 0 else None

        class_rows.append({
            "label": label,
            "support": support,
            "correct": correct,
            "recall": recall
        })
        # <<< A10
    # 클래스별 평가 결과를 CSV 파일로 저장합니다.
    pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)

    # 각 검증 데이터의 실제 정답과 모델 예측 결과를 CSV로 저장합니다.
    pd.DataFrame({"id":valid["id"].tolist(),"actual":labels.tolist(),
                  "predicted":best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)

    # 요약표에 복원된 모델의 평가 결과를 추가합니다.
    summary[-1]["restored_valid_loss"] = restored_loss
    summary[-1]["restored_matches_best"] = abs(restored_loss-best_loss) < 1e-6

    # Epoch에 따른 Train Loss와 Valid Loss의 변화를 그래프로 표시합니다.
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")

    # 최적 Epoch 위치에 점선을 표시합니다.
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    # 실험별 Loss 그래프를 PNG로 저장합니다.
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    plt.close()

    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))

result = pd.DataFrame(summary)
result.to_csv(out / "07_comparison.csv", index=False)
print("생성: outputs/07_*.csv, *.png / test 미사용")
