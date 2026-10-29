"""실습 04: 같은 파라미터의 Module과 Sequential을 비교합니다."""

import torch
from torch import nn  # 신경망을 구성하는 클래스와 함수들을 가져옵니다.


# 사용자 정의 신경망 클래스를 만듭니다.
# nn.Module은 PyTorch에서 신경망 모델을 만들 때 사용하는 기본 클래스입니다.
class SimpleANN(nn.Module):

    def __init__(self):
        # 부모 클래스인 nn.Module의 초기화 작업을 수행합니다.
        super().__init__()

        # 신경망 층을 순서대로 연결합니다.
        # 입력 2개 → 은닉층 8개 → 출력 2개
        self.net = nn.Sequential(
            nn.Linear(2, 8),  # 입력 특성 2개를 받아 8개로 변환합니다.
            nn.ReLU(),        # 음수는 0으로 바꾸는 활성화 함수를 적용합니다.
            nn.Linear(8, 2)   # 8개의 값을 받아 최종 출력 2개를 만듭니다.
        )

    def forward(self, x):
        # 입력 데이터 x를 self.net에 전달해 계산합니다.
        # 결과인 logits에는 각 데이터의 출력값 2개가 들어갑니다.
        # Softmax는 적용하지 않습니다.
        logits = self.net(x)

        # 계산한 모델 출력값을 반환합니다.
        return logits


# 난수 생성 기준을 고정합니다.
# 같은 조건에서 모델을 생성할 때 초기 파라미터를 재현하기 쉽게 합니다.
torch.manual_seed(42)

# 사용자 정의 클래스 SimpleANN으로 모델을 생성합니다.
custom = SimpleANN()

# nn.Sequential만 사용해 동일한 구조의 모델을 생성합니다.
# 입력 2개 → 은닉층 8개 → 출력 2개
sequential = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

# custom 모델의 파라미터를 sequential 모델에 복사합니다.
# 두 모델이 서로 다른 초기 파라미터 때문에 다른 출력을 내는 상황을 방지합니다.
sequential.load_state_dict(custom.net.state_dict())


# 테스트용 입력 데이터 3개를 생성합니다.
# 각 데이터는 특성 2개로 구성되어 있습니다.
X = torch.tensor([
    [1., 2.],   # 첫 번째 데이터
    [0., 1.],   # 두 번째 데이터
    [-1., 0.]   # 세 번째 데이터
])

# 같은 입력 X를 두 모델에 각각 전달합니다.
# custom(X)는 SimpleANN의 forward()를 통해 계산됩니다.
# sequential(X)는 Sequential에 연결된 층을 순서대로 통과합니다.
a, b = custom(X), sequential(X)

# 각 모델의 출력 데이터 형태(shape)를 출력합니다.
# 입력 데이터가 3개이고 각 데이터의 출력값이 2개이므로 (3, 2)가 나옵니다.
print("출력 shape:", tuple(a.shape), tuple(b.shape))

# 두 모델의 출력값이 서로 거의 같은지 확인합니다.
# 부동소수점 연산의 미세한 오차를 허용하며 비교합니다.
# 같으면 True, 다르면 False가 출력됩니다.
print("동일 출력:", torch.allclose(a, b))

# custom 모델이 가지고 있는 학습 가능한 파라미터의 총 개수를 계산합니다.
# 가중치와 편향의 원소 수를 모두 더합니다.
print("파라미터:", sum(p.numel() for p in custom.parameters()))