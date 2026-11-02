"""실습 05 · 제공된 CNN으로 5 epoch 학습한다. test는 읽지 않는다."""
from pathlib import Path
import copy
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 현재 파이썬 파일이 위치한 폴더를 기준 경로로 설정한다.
root = Path(__file__).resolve().parent

# 난수 시드를 42로 고정한다.
# 무작위 작업의 결과를 재현하기 쉽게 만든다.
torch.manual_seed(42)

# PyTorch에서 사용하는 CPU 스레드 수를 1개로 제한한다.
torch.set_num_threads(1)


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


# 학습 데이터 전체를 하나의 배치로 가져온다.
# batch_size=len(train)이므로 학습 이미지 개수만큼 한 번에 가져온다.
# iter()로 DataLoader의 반복자를 만들고 next()로 첫 번째 배치를 가져온다.
x_train, y_train = next(
    iter(DataLoader(
        train,
        batch_size=len(train),
        shuffle=False
    ))
)


# 검증 데이터 전체도 하나의 배치로 가져온다.
# 검증 데이터를 한 번에 모델에 입력해 평가하기 위한 준비 과정이다.
x_valid, y_valid = next(
    iter(DataLoader(
        valid,
        batch_size=len(valid),
        shuffle=False
    ))
)


# CNN 모델을 순서대로 구성한다.
model = nn.Sequential(
    # RGB 3채널 이미지를 입력받아 16개의 특징 맵을 생성한다.
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


# AdamW 최적화 알고리즘을 설정한다.
optimizer = torch.optim.AdamW(
    model.parameters(),  # 모델의 학습 가능한 가중치를 전달한다.
    lr=0.01,              # 학습률을 0.01로 설정한다.
    weight_decay=0.0      # 가중치 감쇠를 적용하지 않는다.
)

# 예측 결과와 실제 정답을 비교해 손실을 계산하는 함수를 설정한다.
loss_fn = nn.CrossEntropyLoss()

# 지금까지 확인한 가장 낮은 검증 손실을 저장할 변수
# float("inf")는 양의 무한대이므로 첫 번째 검증 손실이 더 작게 된다.
best_loss = float("inf")


# 1부터 5까지 총 5 Epoch 동안 학습한다.
for epoch in range(1, 6):

    # 모델을 학습 모드로 전환한다.
    model.train()

    # 이전에 계산한 기울기를 초기화한다.
    # 이번 실습은 전체 데이터를 하나의 배치로 사용하므로
    # Epoch마다 한 번 초기화하면 된다.
    optimizer.zero_grad()

    # 순전파를 수행한다.
    # 전체 학습 이미지를 모델에 입력해 예측 점수(Logits)를 계산한다.
    logits = model(x_train)

    # 예측 점수와 실제 정답을 비교해 손실을 계산한다.
    loss = loss_fn(logits, y_train)


    # TODO BEGIN B05

    # 역전파를 실행해 각 가중치의 기울기를 계산한다.
    loss.backward()

    # 계산된 기울기를 이용해 모델의 가중치를 갱신한다.
    optimizer.step()

    # END B05


    # 학습이 끝난 모델을 평가 모드로 전환한다.
    model.eval()

    # 평가 단계에서는 기울기를 계산하지 않는다.
    with torch.no_grad():

        # 학습 데이터에 대한 손실을 다시 계산한다.
        # 학습 후 갱신된 모델의 성능을 확인한다.
        train_loss = loss_fn(
            model(x_train),
            y_train
        ).item()

        # 검증 이미지를 모델에 입력해 예측 점수를 계산한다.
        valid_logits = model(x_valid)

        # 검증 데이터의 손실을 계산한다.
        # .item()은 텐서의 단일 값을 Python 숫자로 변환한다.
        valid_loss = loss_fn(
            valid_logits,
            y_valid
        ).item()

        # 가장 높은 예측 점수의 클래스 인덱스를 선택한다.
        # 실제 정답과 비교해 평균 정확도를 계산한다.
        valid_acc = (
            valid_logits.argmax(1) == y_valid
        ).float().mean().item()


    # 현재 Epoch의 학습 손실, 검증 손실, 검증 정확도를 출력한다.
    # :.4f는 소수점 넷째 자리까지 표시한다.
    # :.2f는 소수점 둘째 자리까지 표시한다.
    print(
        f"epoch={epoch} "
        f"train_loss={train_loss:.4f} "
        f"valid_loss={valid_loss:.4f} "
        f"valid_acc={valid_acc:.2f}"
    )


    # 현재 검증 손실이 지금까지의 최저 손실보다 작으면
    # 현재 모델을 가장 좋은 모델로 판단한다.
    if valid_loss < best_loss:

        # 가장 낮은 검증 손실과 해당 Epoch 번호를 저장한다.
        best_loss, best_epoch = valid_loss, epoch

        # 현재 모델의 가중치를 독립적인 복사본으로 저장한다.
        # 이후 학습으로 가중치가 변경되어도 저장한 값은 유지된다.
        best_state = copy.deepcopy(model.state_dict())


# 5 Epoch 학습이 끝난 뒤 가장 성능이 좋았던 가중치를 불러온다.
model.load_state_dict(best_state)

# 선택한 모델의 Epoch 번호를 출력한다.
print("선택한 epoch:", best_epoch)