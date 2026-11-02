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

"""
[Downsampling 관련 개념]

Downsampling(다운샘플링): 데이터의 공간 크기를 줄이는 작업이다.
이미지의 높이와 너비를 줄여 연산량과 데이터 크기를 감소시킬 수 있다.

MaxPool2d(최대 풀링): 일정한 영역에서 가장 큰 값을 선택하는 연산이다.
MaxPool2d(2)는 기본적으로 2×2 영역에서 최댓값을 선택한다.
기본 stride는 커널 크기와 동일하므로 공간 크기가 절반으로 줄어든다.
일반적인 MaxPool2d는 학습 가능한 파라미터가 없다.

Max Pooling(최대 풀링): 주변 영역에서 가장 큰 특징값을 남기는 방법이다.

Conv2d(합성곱): 커널을 이용해 이미지의 특징을 추출하는 연산이다.
stride=2를 사용하면 커널이 두 칸씩 이동하여 출력 공간 크기가 줄어든다.
합성곱 층은 Weight와 Bias라는 학습 가능한 파라미터를 가진다.

Parameters(파라미터): 모델이 학습하는 가중치와 편향의 개수다.
MaxPool2d는 학습 파라미터가 없지만 Conv2d는 학습 파라미터를 가진다.

Downsampling 비교:
MaxPool2d: 최댓값을 선택하여 공간 크기를 줄인다.
Conv2d: 학습 가능한 커널을 사용하여 특징을 추출하면서 공간 크기를 줄인다.
"""

torch.manual_seed(42)
torch.set_num_threads(1)
root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

# 임의의 이미지 특징 데이터를 생성한다.
# Tensor 형태: [4, 8, 32, 32]
# 4: Batch 크기, 이미지 데이터 4개
# 8: 입력 채널 8개
# 32: 높이
# 32: 너비
x = torch.rand(4, 8, 32, 32)

operations = {# 비교할 두 가지 다운샘플링 층을 딕셔너리로 생성한다.
    
    "pool": nn.MaxPool2d(2),    # "pool": 2×2 영역에서 최댓값을 선택하는 MaxPool2d

    # "conv_stride2": 3×3 커널을 사용하고 두 칸씩 이동하는 Conv2d
    # nn.MaxPool2d(2): 기본적으로 2×2 영역의 최댓값을 선택한다.
    # nn.Conv2d(8, 8, 3, stride=2, padding=1):
    # 입력 채널 8개, 출력 채널 8개, 커널 3×3, 이동 간격 2, 패딩 1
    "conv_stride2": nn.Conv2d(8, 8, 3, stride=2, padding=1)
}

rows = []   # 각 다운샘플링 방법의 실험 결과를 저장할 빈 리스트

# 딕셔너리의 이름과 층 객체를 하나씩 가져와 반복한다.
# name: 현재 층의 이름 ("pool" 또는 "conv_stride2")
# layer: 해당 이름에 연결된 실제 신경망 층
for name, layer in operations.items():
    # BEGIN A03
    # TODO A03: 각 downsampling 층을 입력 x에 적용한다.
    # pool이면 최대 풀링을 수행하고,
    # conv_stride2이면 합성곱 연산을 수행한다.
    y = layer(x)
    # END A03
    
    # 현재 실험 결과를 딕셔너리로 만들어 리스트에 추가한다.
    # method: 사용한 다운샘플링 방법의 이름
    # shape: 출력 Tensor의 형태
    # params: 학습 파라미터 개수
        # 현재 층의 학습 파라미터 개수를 계산한다.
        # MaxPool2d는 학습 파라미터가 없으므로 0이 나온다.
        # Conv2d는 Weight와 Bias의 개수가 계산된다.
    # tuple(y.shape): 출력 Tensor의 크기를 튜플로 변환한다.
    # str(): 튜플을 문자열로 변환하여 저장한다.
    rows.append(dict(method=name, shape=str(tuple(y.shape)), params=sum(p.numel() for p in layer.parameters())))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(out / "downsample_comparison.csv", index=False)
