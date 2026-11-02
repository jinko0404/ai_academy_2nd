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
[Stride와 연산량 관련 개념]

Stride(스트라이드): 커널이 이미지 위에서 이동하는 간격이다.
Stride=1: 한 칸씩 이동한다.
Stride=2: 두 칸씩 이동한다.
Stride 값이 커지면 일반적으로 출력 이미지의 공간 크기가 작아진다.

Padding(패딩): 이미지 가장자리에 추가 공간을 넣는 방법이다.
출력 이미지의 공간 크기를 조절하는 데 사용한다.

MAC(Multiply-Accumulate): 곱셈 후 누적하는 연산이다.
1 MAC은 곱셈과 누적을 한 번 수행하는 연산 단위다.
이번 실습에서는 Bias 덧셈을 제외하고 계산한다.

Parameters(파라미터): 모델이 학습하는 가중치와 편향의 전체 개수다.
Stride가 변경되어도 합성곱 층의 파라미터 개수는 변하지 않는다.

DataFrame: Pandas에서 데이터를 표 형태로 관리하는 자료 구조다.
to_string(): DataFrame을 문자열 표 형태로 출력한다.
to_csv(): DataFrame을 CSV 파일로 저장한다.
"""

torch.manual_seed(42)   # 동일한 난수 결과를 재현하기 위한 시드 설정

# PyTorch가 CPU에서 사용할 스레드 개수를 1개로 제한한다.
# Thread(스레드): 프로그램에서 작업을 실행하는 처리 단위다.
torch.set_num_threads(1)

root = Path(__file__).resolve().parent  # 현재 실행 중인 Python 파일이 위치한 폴더 경로를 가져온다.
out = root / "outputs"                  # 결과 파일을 저장할 outputs 폴더 경로를 지정한다.
out.mkdir(exist_ok=True)                # outputs 폴더가 없으면 생성하며, 폴더가 이미 존재해도 오류를 일으키지 않는다.

# RGB 이미지 4장에 해당하는 임의의 Tensor 데이터를 생성한다.
# 형태: [Batch 4, Channel 3, Height 32, Width 32]
x = torch.rand(4, 3, 32, 32)

rows = []   # Stride별 실험 결과를 저장할 빈 리스트를 생성한다.

for stride in [1, 2]:   # Stride 값을 1과 2로 변경하면서 각각 실험한다.
    torch.manual_seed(42)   # 동일한 난수 결과를 재현하기 위한 시드 설정

    # BEGIN A02
    # TODO A02: 현재 반복의 stride를 Conv2d에 전달한다.
    # 입력 채널 3개, 출력 채널 8개, 커널 크기 3×3인 합성곱 층을 생성한다.
    # stride: 커널의 이동 간격
    # padding=1: 이미지 가장자리에 1칸의 공간을 추가한다.
    conv = nn.Conv2d(3, 8, 3, stride=stride, padding=1)
    # END A02

    y = conv(x) # 생성한 합성곱 층에 이미지 Tensor를 입력하여 특징을 추출한다.

    # 합성곱 층이 가지고 있는 전체 학습 파라미터 개수를 계산한다.
    # conv.parameters(): Weight와 Bias를 가져온다.
    # p.numel(): 각 파라미터 Tensor에 들어 있는 원소 개수를 계산한다.
    # sum(): 모든 파라미터의 개수를 더한다.
    params = sum(p.numel() for p in conv.parameters())

    # MAC(Multiply-Accumulate): 곱셈 후 누적하는 연산 횟수다.
    # y.numel(): 출력 Tensor의 전체 원소 개수를 구한다.
    # (3 * 3 * 3): 출력 원소 하나를 계산할 때 필요한 곱셈·누적 연산 수다.
    # 커널 높이 3 × 커널 너비 3 × 입력 채널 3
    # 출력 원소 전체 개수에 곱하여 Batch 전체의 MACs를 계산한다.
    # Bias 덧셈은 계산에서 제외한다.
    macs = y.numel() * (3 * 3 * 3)

    # 현재 Stride의 실험 결과를 딕셔너리 형태로 리스트에 추가한다.
    # stride: 현재 사용한 Stride 값
    # shape: 출력 Tensor의 형태
    # params: 학습 파라미터 개수
    # macs: 전체 MAC 연산량
    # dict(): 데이터를 키와 값으로 구성된 딕셔너리로 만든다.
    # tuple() 형태의 출력 크기를 str()로 문자열로 변환하여 저장한다.
    rows.append(dict(
        stride=stride,
        shape=str(tuple(y.shape)),
        params=params,
        macs=macs
    ))

# 실험 결과가 저장된 리스트를 Pandas DataFrame으로 변환한다.
# to_string(): 표 형태의 데이터를 문자열로 변환하여 터미널에 출력한다.
print(pd.DataFrame(rows).to_string(index=False))

# 실험 결과를 CSV 파일로 저장한다.
# CSV: 표 형태의 데이터를 저장하는 파일 형식이다.
# index=False: DataFrame의 행 번호를 별도 열로 저장하지 않는다.
pd.DataFrame(rows).to_csv(out / "stride_comparison.csv", index=False)
