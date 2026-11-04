import torch
from torch import nn

"""
============================================================
[개념 정리] Padding Mask와 Attention
============================================================

1. Padding (패딩)
   - 길이가 서로 다른 데이터를 동일한 크기로 맞추기 위해 추가하는 빈 데이터
   - 일반적으로 0으로 채운다.
   - 실제 정보가 아니므로 모델이 의미 있는 데이터로 학습하거나 참고하지 않도록 해야 한다.

2. Sequence Length (시퀀스 길이)
   - 하나의 데이터가 실제로 가지고 있는 데이터의 개수
   - 여기서는 첫 번째 데이터가 3개, 두 번째 데이터가 5개이다.

3. Padding Mask (패딩 마스크)
   - Padding이 위치한 곳을 표시하는 마스크
   - 실제 데이터와 Padding 영역을 구분한다.
   - True: Padding 영역
   - False: 실제 데이터 영역

4. key_padding_mask
   - MultiheadAttention에서 Key의 Padding 위치를 제외하는 기능
   - True로 표시된 Key 위치는 Attention 계산에서 제외된다.
   - 입력 형태는 [Batch, Sequence]이다.

5. Boolean Mask (불리언 마스크)
   - True 또는 False로 구성된 마스크
   - 조건에 따라 특정 위치를 선택하거나 제외할 때 사용한다.

6. Query와 Key의 차이
   - Query: 어떤 정보를 찾고 있는지 나타내는 값
   - Key: 찾고자 하는 정보와 비교하는 대상
   - key_padding_mask는 Key 위치를 제외하지만 Query 위치까지 자동으로 제거하지는 않는다.

7. Attention Weight (어텐션 가중치)
   - 각 Query가 각각의 Key에 얼마나 집중하는지 나타내는 값
   - 제외된 Key 위치의 가중치는 0이 된다.
   - 각 Query의 가중치 합은 1이다.

8. Masked Mean (마스킹 평균)
   - Padding 영역을 제외하고 실제 데이터만 평균을 계산하는 방법
   - Padding 위치가 평균 결과에 영향을 주지 않도록 한다.

9. unsqueeze()
   - 기존 텐서에 크기가 1인 차원을 추가하는 함수
   - 차원을 맞춰 연산할 때 사용한다.

10. Broadcasting (브로드캐스팅)
    - 크기가 다른 텐서끼리 연산할 수 있도록 차원을 자동으로 맞추는 기능
    - 여기서는 마스크를 특징 차원에 맞춰 적용한다.
"""

torch.set_num_threads(1)
# CPU 연산에 사용할 스레드 수를 1개로 고정한다.

torch.manual_seed(42)
# 난수 시드를 고정하여 동일한 난수 결과를 재현한다.

from pathlib import Path
# 파일 및 폴더 경로를 편리하게 처리하기 위한 모듈

import pandas as pd
# 표 형태의 데이터 처리 및 분석을 위한 라이브러리
# 현재 코드에서는 직접 사용하지 않는다.

# 현재 파이썬 파일이 위치한 폴더를 기준 경로로 설정한다.
ROOT = Path(__file__).resolve().parent

# 결과물을 저장할 outputs 폴더 경로를 지정한다.
OUT = ROOT / "outputs"

# outputs 폴더가 없으면 생성한다.
# exist_ok=True: 폴더가 이미 있어도 오류가 발생하지 않는다.
OUT.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. 시각화 환경 설정
# ------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
# 화면에 그래프를 직접 표시하지 않고 이미지 파일로 저장하는 방식

import matplotlib.pyplot as plt
# 그래프 및 시각화를 위한 라이브러리


# ------------------------------------------------------------
# 2. 입력 데이터와 Padding Mask 생성
# ------------------------------------------------------------

x = torch.randn(2, 5, 16)
# 입력 데이터 생성
#
# (2, 5, 16)
# 2: Batch Size (데이터 묶음 개수)
# 5: Sequence Length (최대 데이터 길이)
# 16: Feature Dimension (특징 차원)

lengths = torch.tensor([3, 5])
# 각 데이터의 실제 길이를 저장한다.
# 첫 번째 데이터: 실제 길이 3
# 두 번째 데이터: 실제 길이 5
#
# 두 데이터의 길이가 다르기 때문에
# 짧은 데이터는 Padding으로 길이를 맞춘다.


# ------------------------------------------------------------
# 3. Padding Mask 생성
# ------------------------------------------------------------

mask = torch.arange(5).unsqueeze(0) >= lengths.unsqueeze(1)
# torch.arange(5): [0, 1, 2, 3, 4] 생성
#
# unsqueeze(0): 행 방향으로 차원 추가
# 결과 형태: (1, 5)
#
# lengths.unsqueeze(1): 열 방향으로 차원 추가
# 결과 형태: (2, 1)
#
# 두 텐서를 비교하여 실제 길이를 초과하는 위치를 True로 표시한다.
#
# 결과:
# 첫 번째 데이터: [False, False, False, True, True]
# 두 번째 데이터: [False, False, False, False, False]
#
# True = Padding
# False = 실제 데이터


x[mask] = 0
# Mask가 True인 위치의 입력 데이터를 0으로 변경한다.
# 즉, Padding 영역을 0으로 채운다.


# ------------------------------------------------------------
# 4. Multi-Head Attention 생성
# ------------------------------------------------------------

attention = nn.MultiheadAttention(
    16,
    4,
    dropout=0,
    batch_first=True
).eval()

# 16: 입력 특징 차원
# 4: Attention Head 개수
# dropout=0: Dropout을 적용하지 않는다.
# batch_first=True: 입력 차원을 (Batch, Sequence, Feature)로 사용한다.
#
# eval(): 평가 모드로 전환한다.
# Dropout이나 Batch Normalization 등이 있는 경우 평가 모드로 동작한다.


# ------------------------------------------------------------
# 5. Attention 계산
# ------------------------------------------------------------

with torch.inference_mode():
    # 추론 모드로 실행한다.
    # 역전파를 위한 그래프를 생성하지 않아 메모리를 절약한다.

    plain, _ = attention(x, x, x)
    # Padding Mask를 적용하지 않은 일반 Attention 계산
    # Q, K, V에 동일한 입력 x를 전달한다.
    # _는 사용하지 않을 Attention Weight를 의미한다.

    # TODO A05: Padding을 key로 보지 않게 bool Mask를 전달한다.
    # nn.MultiheadAttention의 key_padding_mask: [B,T], True가 제외 위치이다.
    # BEGIN A05
    masked, weights = attention(
        x, x, x,
        key_padding_mask=mask,
        average_attn_weights=False
    )
    # END A05

    # key_padding_mask는 Padding 위치를 Key에서 제외한다.
    # 하지만 Padding 위치의 Query 출력까지 0으로 만들지는 않는다.
    # 따라서 평균을 계산할 때 Padding Query를 다시 제외해야 한다.

    mean = (masked * (~mask).unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
    # ~mask: 마스크 값을 반전한다.
    # True였던 Padding 위치는 False가 된다.
    # False였던 실제 데이터 위치는 True가 된다.
    #
    # unsqueeze(-1): 마지막에 특징 차원 추가
    # 결과 형태: (2, 5, 1)
    #
    # masked와 곱하여 실제 데이터 위치만 남긴다.
    #
    # sum(1): Sequence 방향으로 합계를 계산한다.
    #
    # lengths.unsqueeze(-1):
    # 각 데이터의 실제 길이를 특징 차원에 맞게 변환한다.
    #
    # 실제 데이터 개수로 나누어 Padding을 제외한 평균을 계산한다.


# ------------------------------------------------------------
# 6. 결과 확인
# ------------------------------------------------------------

print("shape:", tuple(masked.shape), tuple(weights.shape))
# masked: Attention 계산 후 출력 데이터의 형태
# 예상 결과: (2, 5, 16)
#
# weights: Attention 가중치의 형태
# 평균 Attention Weight이므로 (2, 5, 5)


print("막은 key weight 최대:", round(weights[0, :, 3:].abs().max().item(), 6))
# 첫 번째 Batch의 Padding Key 위치(3, 4)의 가중치를 확인한다.
# abs(): 절댓값 계산
# max(): 가장 큰 값 확인
# item(): 텐서에서 숫자 하나 추출
# round(): 소수점 6자리까지 반올림
#
# 정상적으로 Padding Key를 제외했다면 결과는 0이다.


print("유효 query 출력 변화:", round((plain[0, :3] - masked[0, :3]).abs().max().item(), 6))
# 첫 번째 데이터의 실제 Query 위치(0~2)를 비교한다.
# 일반 Attention과 Padding Mask 적용 Attention의 출력 차이를 확인한다.
#
# Padding Key가 제외되었기 때문에 유효 Query의 출력이 달라진다.


print("masked mean:", tuple(mean.shape))
# Padding을 제외하고 평균을 계산한 결과의 형태
# 예상 결과: (2, 16)


# ------------------------------------------------------------
# 7. Padding Mask 시각화
# ------------------------------------------------------------

plt.figure(figsize=(6, 3))
# 그래프 크기를 가로 6, 세로 3으로 설정한다.

plt.imshow(mask.int().numpy(), vmin=0, vmax=1)
# Mask를 정수형으로 변환하여 이미지로 표시한다.
# False = 0 (실제 데이터)
# True = 1 (Padding)

plt.xlabel("Time")
# X축 이름 설정

plt.ylabel("Sample")
# Y축 이름 설정

plt.title("Padding mask: 1 = ignored")
# 그래프 제목 설정

plt.xticks(range(5))
# X축 눈금을 0~4로 설정한다.

plt.yticks(range(2))
# Y축 눈금을 0~1로 설정한다.

plt.tight_layout()
# 그래프 요소가 겹치지 않도록 여백을 조정한다.

plt.savefig(OUT / "mask_or_latent.png", dpi=120)
# outputs 폴더에 그래프 이미지를 저장한다.
# dpi=120: 이미지 해상도 설정

plt.close()
# 그래프를 닫아 메모리를 정리한다.
