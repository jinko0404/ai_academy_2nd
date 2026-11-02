"""
실습 A05: CNN 모델 학습 및 평가

[새로운 개념]

1. Forward(순전파)
   - 입력 데이터를 모델에 전달해 예측 결과를 계산하는 과정
   - model(images)를 실행하면 예측 점수(Logits)가 출력된다.

2. Loss(손실)
   - 모델의 예측 결과와 실제 정답의 차이를 수치로 계산한 값
   - 손실이 작을수록 실제 정답에 가까운 예측을 했다는 의미다.

3. CrossEntropyLoss()
   - 분류 모델에서 사용하는 손실 함수
   - 모델의 예측 점수와 실제 정답을 비교해 손실을 계산한다.
   - 다중 클래스 분류에도 사용할 수 있다.

4. Backpropagation(역전파)
   - 손실을 바탕으로 각 가중치의 기울기를 계산하는 과정
   - loss.backward()로 실행한다.

5. Gradient(기울기)
   - 가중치를 어느 방향으로 얼마나 변경해야 하는지 알려주는 값
   - 손실을 줄이기 위한 가중치 수정에 사용된다.

6. Optimizer(최적화 도구)
   - 계산된 기울기를 이용해 모델의 가중치를 갱신하는 도구
   - optimizer.step()으로 가중치를 수정한다.

7. AdamW
   - 모델 학습에 사용하는 최적화 알고리즘
   - 기울기를 바탕으로 가중치를 갱신하며 가중치 감쇠 기능도 제공한다.

8. Learning Rate(lr)
   - 가중치를 한 번 갱신할 때 변경하는 크기를 조절하는 값
   - 값이 너무 크면 학습이 불안정해질 수 있고, 너무 작으면 학습이 느려질 수 있다.

9. Weight Decay
   - 가중치가 지나치게 커지는 것을 억제하기 위한 정규화 기법
   - AdamW에서는 weight_decay로 설정한다.

10. Epoch
    - 전체 학습 데이터를 한 번 모두 학습한 단위
    - 이 코드에서는 총 5회 학습한다.

11. model.train()
    - 모델을 학습 모드로 전환한다.

12. model.eval()
    - 모델을 평가 모드로 전환한다.
    - Dropout이나 BatchNorm과 같은 계층의 동작 방식이 평가에 맞게 변경된다.

13. torch.no_grad()
    - 해당 구간에서 기울기를 계산하지 않도록 설정한다.
    - 평가 단계에서 불필요한 기울기 계산을 줄일 수 있다.

14. torch.argmax()
    - 가장 큰 값의 인덱스를 반환한다.
    - 분류 모델의 예측 클래스를 선택할 때 사용한다.

15. state_dict()
    - 모델의 학습 가능한 가중치 등 상태 정보를 저장하는 기능

16. copy.deepcopy()
    - 원본과 독립적인 복사본을 생성한다.
    - 가장 성능이 좋았던 모델의 가중치를 별도로 보관할 때 사용한다.

17. Confusion Matrix(혼동행렬)
    - 실제 정답과 모델의 예측 결과를 비교한 표
    - 행은 실제 클래스, 열은 예측 클래스로 구성할 수 있다.

18. torch.zeros()
    - 지정한 크기와 자료형으로 0으로 채워진 텐서를 생성한다.

19. diag()
    - 행렬의 대각선 원소를 가져온다.
    - 혼동행렬에서는 클래스별 정답 수를 의미한다.

20. zip()
    - 여러 반복 가능한 자료에서 같은 순서의 원소를 묶어준다.

21. DataFrame()
    - Pandas에서 표 형태의 데이터를 생성하는 기능

22. to_csv()
    - DataFrame 데이터를 CSV 파일로 저장하는 기능

23. .item()
    - 텐서에 들어 있는 단일 값을 일반 Python 숫자로 변환한다.

24. .tolist()
    - 텐서의 값을 Python 리스트로 변환한다.


[전체 학습 흐름]

1. 이미지 데이터 불러오기
2. DataLoader로 데이터를 배치 단위로 구성
3. CNN 모델 구성
4. 모델의 예측 결과 계산(순전파)
5. 손실 계산
6. 역전파로 기울기 계산
7. Optimizer로 가중치 갱신
8. 검증 데이터로 성능 평가
9. 검증 손실이 가장 낮은 모델 저장
10. 최적 모델로 혼동행렬과 예측 결과 생성
11. 학습 기록과 평가 결과 CSV 저장
"""
from pathlib import Path
import copy
import csv
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

torch.manual_seed(42)                   # 난수 시드를 42로 고정한다.
torch.set_num_threads(1)                # PyTorch에서 사용하는 CPU 스레드 수를 1개로 제한한다.
root = Path(__file__).resolve().parent  # 현재 파이썬 파일이 위치한 폴더를 기준 경로로 설정한다.
out = root / "outputs"                  # 결과물을 저장할 outputs 폴더 경로를 지정한다.
out.mkdir(exist_ok=True)                # outputs 폴더가 존재하지 않으면 생성한다.

# 이미지 크기를 32x32로 변경하고 텐서로 변환하는 작업을 묶는다.
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 이미지 크기를 32x32로 변경한다.
    transforms.ToTensor()          # 이미지를 PyTorch 텐서로 변환한다.
])

# ImageFolder를 이용해 학습 이미지 데이터를 불러온다.
# 하위 폴더 이름을 기준으로 클래스를 자동으로 구분한다.
train = datasets.ImageFolder(
    root / "data" / "train",
    transform=transform
)

# 검증 이미지 데이터를 불러온다.
# 학습 데이터와 동일한 이미지 변환을 적용한다.
valid = datasets.ImageFolder(
    root / "data" / "valid",
    transform=transform
)

# 학습 데이터를 배치 단위로 제공하는 DataLoader를 생성한다.
train_loader = DataLoader(
    train,
    batch_size=16,  # 한 번에 이미지 16개를 모델에 전달한다.
    shuffle=True,   # 학습할 때 데이터 순서를 무작위로 섞는다.
    num_workers=0,  # 별도의 데이터 로딩 작업 프로세스를 사용하지 않는다.
    generator=torch.Generator().manual_seed(42)  # 데이터 섞기에 사용할 난수 시드를 고정한다.
)

# 검증 데이터를 배치 단위로 제공하는 DataLoader를 생성한다.
valid_loader = DataLoader(
    valid,
    batch_size=16,  # 한 번에 이미지 16개씩 처리한다.
    shuffle=False,  # 검증 데이터 순서는 섞지 않는다.
    num_workers=0
)

# CNN 모델을 순서대로 구성한다.
model = nn.Sequential(
    # 입력 이미지: (배치, 3, 32, 32)
    # RGB 3채널을 입력받아 16개의 특징 맵을 생성한다.
    nn.Conv2d(3, 16, 3, padding=1),

    nn.ReLU(),  # 음수 값을 0으로 바꾸는 활성화 함수

    nn.MaxPool2d(2),  # 특징 맵의 가로와 세로 크기를 절반으로 줄인다.

    # 16개 특징 맵을 입력받아 32개 특징 맵을 생성한다.
    nn.Conv2d(16, 32, 3, padding=1),

    nn.ReLU(),  # 활성화 함수 적용

    nn.MaxPool2d(2),  # 특징 맵의 가로와 세로 크기를 다시 절반으로 줄인다.

    # 각 특징 맵의 공간 크기를 1x1로 압축한다.
    nn.AdaptiveAvgPool2d(1),

    # 배치 차원은 유지하고 나머지 차원을 1차원으로 펼친다.
    nn.Flatten(),

    # 32개의 특징을 입력받아 2개 클래스의 예측 점수를 출력한다.
    nn.Linear(32, 2),
)

loss_fn = nn.CrossEntropyLoss() # 예측 결과와 실제 정답을 비교해 손실을 계산하는 함수를 설정한다.

# AdamW 최적화 알고리즘을 설정한다.
optimizer = torch.optim.AdamW(
    model.parameters(),  # 모델의 학습 가능한 가중치를 전달한다.
    lr=0.01,              # 학습률을 0.01로 설정한다.
    weight_decay=0.0      # 가중치 감쇠를 적용하지 않는다.
)

history = []    # Epoch별 학습 및 평가 결과를 저장할 리스트

# 지금까지 확인한 가장 낮은 검증 손실을 저장한다.
# float("inf")는 무한대를 의미한다.
# 첫 번째 검증 손실은 무한대보다 작으므로 최초 모델이 저장된다.
best_loss = float("inf")

for epoch in range(1, 6):   # 총 5번의 Epoch 동안 모델을 학습한다.
    model.train()           # 모델을 학습 모드로 전환한다.
    
    for images, labels in train_loader: # 학습 DataLoader에서 이미지와 정답을 배치 단위로 가져온다.
        # 이전 배치에서 계산한 기울기를 초기화한다.
        # PyTorch는 기본적으로 기울기를 누적하기 때문에 필요하다.
        optimizer.zero_grad()

        loss = loss_fn(model(images), labels)   # 순전파를 수행해 예측값을 계산하고 손실을 구한다.

        # BEGIN A05
        # TODO A05: loss 역전파와 가중치 갱신을 순서대로 실행한다.
        # 역전파를 실행해 각 가중치의 기울기를 계산한다.
        loss.backward()

        # 계산된 기울기를 이용해 모델의 가중치를 갱신한다.
        optimizer.step()
        # END A05
    # 같은 epoch 말 모델을 eval 모드로 평가해 train/valid의 측정 조건을 맞춘다.
    # 한 Epoch의 학습이 끝나면 평가 모드로 전환한다.
    # 학습 모드와 평가 모드의 동작 차이를 반영해 성능을 측정한다.
    model.eval()

    row = {"epoch": epoch}  # 현재 Epoch의 결과를 저장할 딕셔너리를 생성한다.

    with torch.no_grad():   # 평가 단계에서는 기울기를 계산하지 않는다.
        # 학습 데이터와 검증 데이터를 각각 평가한다.
        # 학습 데이터 평가용 DataLoader도 별도로 생성한다.
        for phase, loader in [
            ("train", DataLoader(train, batch_size=16)),
            ("valid", valid_loader)
        ]:
            total_loss, correct, count = 0.0, 0, 0 # 손실 합계, 정답 개수, 전체 데이터 개수를 초기화한다.

            for images, labels in loader:    # 각 DataLoader에서 이미지와 정답을 배치 단위로 가져온다.
                logits = model(images)  # 학습된 모델에 이미지를 입력해 예측 점수(Logits)를 계산한다.

                # 현재 배치의 평균 손실에 데이터 개수를 곱해 손실 합계를 누적한다.
                # .item()은 텐서의 단일 값을 Python 숫자로 변환한다.
                total_loss += loss_fn(logits, labels).item() * len(labels)

                # 가장 높은 예측 점수의 클래스 인덱스를 가져온다.
                # 실제 정답과 비교해 맞힌 개수를 누적한다.
                correct += (logits.argmax(1) == labels).sum().item()

                # 현재 배치에 포함된 데이터 개수를 누적한다.
                count += len(labels)

            row[phase+"_loss"] = total_loss/count    # 전체 데이터의 평균 손실을 계산한다.
            row[phase+"_accuracy"] = correct/count   # 전체 데이터의 정확도를 계산한다.(정답 개수 / 전체 데이터 개수)

    history.append(row) # 현재 Epoch의 평가 결과를 history 리스트에 추가한다.

    # 현재 검증 손실이 이전에 기록한 가장 낮은 손실보다 작으면
    # 현재 모델을 가장 좋은 모델로 판단한다.
    if row["valid_loss"] < best_loss:
         # 가장 낮은 검증 손실과 해당 Epoch 번호를 저장한다.
        best_loss, best_epoch = row["valid_loss"], epoch

        # 현재 모델의 가중치를 독립적인 복사본으로 저장한다.
        # 이후 학습으로 모델 가중치가 바뀌어도 저장한 값은 유지된다.
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)   # 모든 Epoch의 학습이 끝난 뒤 가장 성능이 좋았던 가중치를 불러온다.
model.eval()                        # 모델을 평가 모드로 전환한다.
actual, predicted = [], []          # 실제 정답과 모델의 예측 결과를 저장할 리스트를 생성한다.

with torch.no_grad():                   # 최종 예측 과정에서는 기울기를 계산하지 않는다.
    for images, labels in valid_loader: # 검증 데이터 전체를 배치 단위로 순회한다.
        actual.extend(labels.tolist())  # 실제 정답을 Python 리스트에 추가한다.

        # 모델의 예측 점수에서 가장 높은 클래스 인덱스를 선택한다.
        # 선택한 예측 결과를 Python 리스트에 추가한다.
        predicted.extend(model(images).argmax(1).tolist())

# 2개 클래스의 혼동행렬을 생성한다.
# 2행 2열이며 모든 값을 0으로 초기화한다.
# 자료형은 정수형(long)으로 설정한다.
cm = torch.zeros(2, 2, dtype=torch.long)

for truth, guess in zip(actual, predicted): # 실제 정답(truth)과 예측 결과(guess)를 순서대로 묶어서 반복한다.
    # 혼동행렬의 행에는 실제 정답,
    # 열에는 모델의 예측 결과를 사용한다.
    # 해당 위치의 값을 1 증가시킨다.
    cm[truth, guess] += 1

# Epoch별 학습 및 평가 기록을 DataFrame으로 변환한다.
# 소수점 넷째 자리까지 반올림한 뒤 표 형태로 출력한다.
print(pd.DataFrame(history).round(4).to_string(index=False))

# 가장 낮은 검증 손실을 기록했던 Epoch 번호를 출력한다.
print("best epoch:", best_epoch)

# 혼동행렬을 출력한다.
# 행은 실제 정답, 열은 예측 결과를 의미한다.
print("혼동행렬 행=실제, 열=예측:\n", cm.tolist())

# 혼동행렬의 대각선 값은 정답을 맞힌 개수다.
# diag()는 대각선 원소를 가져오고,
# sum(1)은 각 행의 전체 데이터 개수를 계산한다.
print("클래스별 정답 수:", cm.diag().tolist(), "/", cm.sum(1).tolist())


# Epoch별 학습 및 평가 기록을 CSV 파일로 저장한다.
pd.DataFrame(history).to_csv(
    out / "cnn_history.csv",
    index=False
)

# 혼동행렬을 CSV 파일로 저장한다.
# 행과 열의 이름에는 학습 데이터의 클래스 이름을 사용한다.
pd.DataFrame(
    cm.numpy(),
    index=train.classes,
    columns=train.classes
).to_csv(out / "confusion_matrix.csv")


# 검증 이미지의 파일 이름, 실제 정답, 예측 결과를 CSV 파일로 저장한다.
pd.DataFrame(
    dict(
        # 검증 데이터에 사용된 이미지 파일의 이름만 가져온다.
        file=[Path(p).name for p, _ in valid.samples],

        # 실제 정답을 저장한다.
        actual=actual,

        # 모델이 예측한 결과를 저장한다.
        predicted=predicted
    )
).to_csv(
    out / "cnn_predictions.csv",
    index=False
)
