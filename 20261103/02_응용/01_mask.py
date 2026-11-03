"""
실습 A01: Padding을 제외한 실제 데이터 평균 계산

[핵심 개념]

1. Sequence(시퀀스)
   - 순서가 있는 데이터의 연속된 묶음이다.
   - 예: 시간에 따라 수집한 센서 데이터

2. Padding(패딩)
   - 서로 길이가 다른 데이터를 하나의 묶음으로 처리하기 위해
     짧은 데이터 뒤에 값을 추가하여 길이를 맞추는 작업이다.
   - 여기서는 부족한 부분을 0으로 채운다.
   - 주의: 0이 실제 데이터일 수도 있으므로 값이 0인지로
     Padding 여부를 판단하면 안 된다.

3. Padding Mask(패딩 마스크)
   - 실제 데이터와 Padding 위치를 구분하는 표시이다.
   - True  : Padding 위치
   - False : 실제 데이터 위치

4. 평균 계산
   - Padding을 제외한 실제 데이터만 합산한다.
   - 합산한 값을 실제 데이터의 길이로 나눈다.

[이번 실습의 목표]
Padding으로 추가된 값을 평균 계산에서 제외하고
각 자료의 실제 데이터 평균을 구한다.
"""

import torch
from torch import nn

# 작은 CPU 실습의 실행 조건을 고정한다.
torch.set_num_threads(1)
torch.manual_seed(42)

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

from torch.nn.utils.rnn import pad_sequence

# --------------------------------------------------
# 1. 서로 길이가 다른 데이터 생성
# --------------------------------------------------

# 각 데이터의 형태는 [시점, 특성]이다.
# 4개 시점, 특성 3개 / 2개 시점, 특성 3개 / 5개 시점, 특성 3개
sequences = [
    torch.ones(4, 3),
    torch.ones(2, 3) * 2,
    torch.ones(5, 3) * 3
]

# 각 자료의 실제 길이를 저장한다.
# 첫 번째 자료는 4개, 두 번째는 2개, 세 번째는 5개이다.
lengths = torch.tensor([4, 2, 5])

# --------------------------------------------------
# 2. Padding 적용
# --------------------------------------------------

# pad_sequence:
# 서로 길이가 다른 데이터를 가장 긴 길이에 맞춰준다.
#
# batch_first=True:
# 데이터의 차원 순서를 [자료 개수, 시점, 특성]으로 만든다.
#
# 결과 형태: [3, 5, 3]
# 자료 3개, 최대 시점 5개, 특성 3개
padded = pad_sequence(sequences, batch_first=True)

# --------------------------------------------------
# 3. Padding Mask 생성
# --------------------------------------------------

# torch.arange(padded.size(1))
# 시점 번호를 0부터 4까지 생성한다.
#
# unsqueeze(0)
# 1차원 데이터를 2차원으로 바꾼다.
# 형태: [1, 5]
#
# lengths.unsqueeze(1)
# 실제 길이 [4, 2, 5]를 세로 형태로 바꾼다.
# 형태: [3, 1]
#
# >= 비교:
# 시점 번호가 실제 길이 이상이면 Padding이다.
#
# 결과 mask 형태: [3, 5]
# True  = Padding
# False = 실제 데이터

mask = torch.arange(padded.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)

# --------------------------------------------------
# 4. Padding을 제외한 평균 계산
# --------------------------------------------------

# ① 실제 데이터 위치만 True로 바꾼다.
# ~ 기호는 True와 False를 반대로 뒤집는다.
# 기존 mask: True = Padding
# valid_mask: True = 실제 데이터
valid_mask = ~mask

# ② 차원을 하나 추가한다.
# 기존 형태: [3, 5]
# 변경 형태: [3, 5, 1]
#
# 마지막 차원을 추가하는 이유:
# padded 데이터의 형태는 [3, 5, 3]이다.
# 특성 3개에 마스크를 적용할 수 있도록 형태를 맞춘다.
valid_mask = valid_mask.unsqueeze(-1)

# ③ 실제 데이터만 남긴다.
# 실제 데이터 위치는 True이므로 값을 유지한다.
# Padding 위치는 False이므로 0으로 만든다.
valid_data = padded * valid_mask

# ④ 실제 데이터들을 합산한다.
# sum(dim=1)은 두 번째 차원인 시점 방향으로 합산한다.
# 결과 형태: [3, 3]
total = valid_data.sum(dim=1)

# ⑤ 실제 데이터 길이로 나눠 평균을 계산한다.
#
# lengths.unsqueeze(1)
# [3] 형태의 길이 정보를 [3, 1] 형태로 변경한다.
#
# 각 자료의 합계를 실제 길이로 나눈다.
# 결과 형태: [3, 3]
mean = total / lengths.unsqueeze(1)

# --------------------------------------------------
# 5. 결과 출력 및 저장
# --------------------------------------------------

print("Padding mask:\n", mask)
print("실제 값 평균:\n", mean)

# Padding Mask를 정수 형태로 변환하여 CSV 파일로 저장한다.
# False = 0, True = 1
pd.DataFrame(mask.int().numpy()).to_csv(
    OUT / "padding_mask.csv",
    index=False
)