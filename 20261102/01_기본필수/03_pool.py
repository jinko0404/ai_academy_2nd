"""실습 03 · 클래스 정의 없이 Sequential로 작은 CNN을 연결한다."""
import torch
from torch import nn

"""
[CNN 구성 관련 개념]

CNN(Convolutional Neural Network): 합성곱 연산을 이용해 이미지의 특징을 추출하는 신경망이다.

Sequential(시퀀셜): 여러 신경망 층을 순서대로 연결하는 기능이다.
입력 데이터가 첫 번째 층부터 마지막 층까지 순서대로 전달된다.
직접 클래스를 정의하지 않고도 간단한 신경망을 구성할 수 있다.

Conv2d(2차원 합성곱): 커널을 이용해 이미지의 특징을 추출하는 층이다.
입력 채널과 출력 채널, 커널 크기 등을 설정한다.

ReLU(Rectified Linear Unit): 음수 값을 0으로 바꾸는 활성화 함수다.
양수 값은 그대로 유지한다.
신경망에 비선형성을 추가하는 역할을 한다.

MaxPool2d(최대 풀링): 일정한 영역에서 최댓값을 선택하는 연산이다.
MaxPool2d(2)는 기본적으로 2×2 영역에서 최댓값을 선택하고 공간 크기를 절반으로 줄인다.

AdaptiveAvgPool2d(적응형 평균 풀링): 입력 데이터의 공간 크기와 관계없이
지정한 출력 크기에 맞춰 평균 풀링을 수행한다.
AdaptiveAvgPool2d(1)은 높이와 너비를 각각 1로 만든다.

Flatten(평탄화): 여러 차원으로 구성된 Tensor를 펼치는 기능이다.
Batch 차원은 유지하고 나머지 차원을 하나로 합친다.
Linear 층에 입력하기 위한 형태로 변환할 때 사용한다.

Linear(선형 층): 입력 특징에 가중치와 편향을 적용하여 출력값을 계산하는 층이다.
입력 특징 수와 출력 특징 수를 설정한다.

Activation Function(활성화 함수): 신경망에 비선형성을 추가하는 함수다.
ReLU가 대표적인 활성화 함수다.

Feature Map(특징 맵): 합성곱 연산을 통해 추출된 특징을 담고 있는 데이터다.
"""

torch.manual_seed(42)

# 임의의 RGB 이미지 4장을 생성한다.
# Tensor 형태: [Batch, Channel, Height, Width]
# 4: 이미지 개수
# 3: RGB 입력 채널
# 32: 이미지 높이
# 32: 이미지 너비
x = torch.rand(4, 3, 32, 32)

# MaxPool2d(2)는 2×2에서 최댓값을 선택한다. 기본 stride도 2이다.
# BEGIN B03
# TODO B03: 출력 높이·너비를 절반으로 만드는 MaxPool을 작성한다.
# 2×2 영역에서 최댓값을 선택하는 최대 풀링 층을 생성한다.
# 기본 stride가 2이므로 높이와 너비가 절반으로 줄어든다.
pool = nn.MaxPool2d(2)
# END B03

# Sequential: 여러 신경망 층을 순서대로 연결하여 하나의 모델로 만든다.
# 입력 데이터 x가 아래에 작성된 층들을 순서대로 통과한다.
model = nn.Sequential(

    # 첫 번째 합성곱 층
    # 입력 채널 3개(RGB)를 받아 출력 특징 채널 16개를 생성한다.
    # 3×3 커널과 padding=1을 사용하여 공간 크기를 유지한다.
    nn.Conv2d(3, 16, 3, padding=1),

    # ReLU 활성화 함수
    # 음수 값은 0으로 변경하고 양수 값은 그대로 유지한다.
    nn.ReLU(),

    # 앞에서 만든 MaxPool 층을 적용한다.
    # 높이와 너비를 32에서 16으로 줄인다.
    pool,


    # 두 번째 합성곱 층
    # 입력 특징 채널 16개를 받아 출력 특징 채널 32개를 생성한다.
    # 3×3 커널과 padding=1을 사용하여 공간 크기를 유지한다.
    nn.Conv2d(16, 32, 3, padding=1),

    # 두 번째 ReLU 활성화 함수
    nn.ReLU(),

    # 두 번째 최대 풀링 층
    # 높이와 너비를 16에서 8로 줄인다.
    nn.MaxPool2d(2),


    # AdaptiveAvgPool2d(1):
    # 높이와 너비를 각각 1로 만들어 출력 공간 크기를 1×1로 맞춘다.
    # 입력 형태 [4, 32, 8, 8] → 출력 형태 [4, 32, 1, 1]
    nn.AdaptiveAvgPool2d(1),

    # Flatten:
    # Batch 차원은 유지하고 나머지 차원을 하나로 펼친다.
    # [4, 32, 1, 1] → [4, 32]
    nn.Flatten(),

    # Linear(32, 2):
    # 입력 특징 32개를 받아 출력값 2개를 생성한다.
    # 마지막 출력은 이미지 분류 등에 사용할 수 있다.
    nn.Linear(32, 2),
)

# 모델에 연결된 각 층을 순서대로 하나씩 가져온다.
# layer에는 현재 실행할 신경망 층이 들어간다.
for layer in model:

    # 현재 층에 입력 Tensor x를 전달하여 연산을 수행한다.
    # 연산 결과를 다시 x에 저장한다.
    # 다음 반복에서는 변경된 x가 다음 층의 입력으로 사용된다.
    x = layer(x)

    # 현재 실행한 층의 이름과 출력 Tensor 형태를 확인한다.
    # type(layer).__name__: 현재 층의 클래스 이름을 가져온다.
    # tuple(x.shape): 현재 층을 통과한 Tensor의 크기를 출력한다.
    print(type(layer).__name__, tuple(x.shape))
