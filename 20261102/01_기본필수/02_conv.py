"""실습 02 · 공간 크기를 유지하고 특징 채널을 8개 만든다."""
import torch
from torch import nn

"""
[Conv2d와 합성곱 개념]

Conv2d: 2차원 합성곱(Convolution)을 수행하는 신경망 층이다.
합성곱: 이미지 위에서 작은 행렬을 이동시키며 특징을 추출하는 연산이다.

nn.Conv2d(입력 채널, 출력 채널, 커널 크기, padding)

Input Channels: 합성곱 층에 입력되는 데이터의 채널 수
Output Channels: 합성곱 연산으로 생성할 특징 채널의 수
Kernel Size: 이미지 위에서 이동하며 특징을 계산하는 작은 행렬의 크기
Padding: 이미지 가장자리에 추가 공간을 넣어 출력 크기를 조절하는 방법

Weight(가중치): 합성곱 연산에 사용되며 학습을 통해 값이 조정되는 요소
Bias(편향): 합성곱 결과에 추가되는 학습 가능한 값

padding=1은 가장자리에 1칸을 추가한다.
기본 stride=1, 커널 크기 3×3과 함께 사용하면 높이와 너비가 유지된다.
"""

torch.manual_seed(42)   # 동일한 난수 결과를 재현하기 위한 시드 설정

# 임의의 이미지 데이터 4장을 생성한다.
# (4, 3, 32, 32)
# 4: 이미지 개수(Batch)
# 3: RGB 입력 채널
# 32: 이미지 높이(Height)
# 32: 이미지 너비(Width)
x = torch.rand(4, 3, 32, 32)

# Conv2d의 필수: 입력 채널, 출력 채널, 커널 크기.
# padding=1은 선택 인자. 3×3, stride=1과 결합하면 높이·너비를 유지한다.
# BEGIN B02
# TODO B02: 3입력·8출력·3×3·padding 1의 합성곱 층을 만든다.
# 입력 채널 3개를 받아 출력 특징 채널 8개를 생성하는 합성곱 층이다.
# 커널 크기는 3×3이며 padding=1을 적용한다.
conv = nn.Conv2d(3, 8, 3, padding=1)
# END B02

# 앞에서 만든 합성곱 층에 이미지 Tensor를 입력한다.
# 입력 이미지의 특징을 추출하여 새로운 특징 Tensor를 생성한다.
y = conv(x)

# 입력 Tensor의 형태를 출력한다.
# 입력 이미지가 몇 장이고 채널 및 공간 크기가 얼마인지 확인한다.
print("입력:", tuple(x.shape))

# 합성곱 연산 이후 출력 Tensor의 형태를 확인한다.
# 입력 채널 3개가 출력 채널 8개로 변경된다.
# Batch 크기와 높이·너비는 유지된다.
print("출력:", tuple(y.shape))

# 합성곱 층이 학습할 전체 파라미터 개수를 계산한다.
# conv.parameters(): 합성곱 층이 가지고 있는 학습 파라미터를 가져온다.
# p.numel(): 각 파라미터 Tensor에 들어 있는 숫자의 개수를 계산한다.
# sum(): 각 파라미터의 개수를 모두 더한다.
print("파라미터:", sum(p.numel() for p in conv.parameters()))
