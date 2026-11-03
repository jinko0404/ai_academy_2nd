"""
실습 B01 · 길이가 다른 시퀀스 데이터를 Padding으로 묶기

[기본 개념 정리]

1. Sequence(시퀀스)
   - 순서가 있는 데이터의 연속적인 묶음.
   - 예: 문장의 단어, 시간에 따른 센서 데이터, 음성 데이터 등.

2. Time Step(시점)
   - 시퀀스에서 데이터가 하나씩 진행되는 각 단계.
   - 예: 5개의 시점이 있다면 데이터가 5단계로 구성되어 있다는 의미.

3. Feature(특성)
   - 각 시점에서 가지고 있는 데이터의 속성 또는 값.
   - 이번 실습에서는 각 시점마다 특성이 3개씩 존재한다.

4. Batch(배치)
   - 여러 개의 데이터를 하나로 묶은 단위.
   - 이번 실습에서는 시퀀스 3개를 하나의 배치로 묶는다.

5. Padding(패딩)
   - 길이가 서로 다른 시퀀스의 길이를 맞추기 위해
     부족한 부분에 특정 값을 추가하는 방법.
   - pad_sequence()의 기본 패딩 값은 0이다.

6. pad_sequence()
   - 길이가 서로 다른 여러 텐서를 하나의 텐서로 묶어주는 함수.
   - 가장 긴 시퀀스의 길이에 맞춰 나머지 시퀀스를 Padding한다.

7. batch_first
   - 최종 텐서에서 Batch 차원을 첫 번째에 배치할지 결정한다.
   - True: [Batch, Sequence Length, Feature]
   - False: [Sequence Length, Batch, Feature]
   - 기본값은 False이다.

8. Tensor Shape(텐서 형태)
   - 텐서가 각 차원에서 얼마나 많은 데이터를 가지고 있는지 나타낸다.
   - 이번 실습의 최종 형태는 [3, 5, 3]이다.
     3: 시퀀스 개수(Batch)
     5: 가장 긴 시퀀스 길이
     3: 각 시점의 특성 개수

[이번 실습의 핵심]

원본 시퀀스 길이:
- 첫 번째 자료: 4
- 두 번째 자료: 2
- 세 번째 자료: 5

가장 긴 시퀀스 길이가 5이므로
나머지 시퀀스에 부족한 만큼 0을 추가한다.
"""


# PyTorch 기본 기능을 가져온다.
import torch

# 신경망 계층을 사용하기 위한 모듈
from torch import nn


# 작은 CPU 실습의 실행 조건을 고정한다.
torch.set_num_threads(1)

# 난수 시드를 고정한다.
torch.manual_seed(42)


# 길이가 다른 시퀀스를 하나의 텐서로 묶는 함수
from torch.nn.utils.rnn import pad_sequence


# 서로 길이가 다른 3개의 시퀀스 데이터를 생성한다.
# 각 시퀀스는 [시점 개수, 특성 개수] 형태를 가진다.

sequences = [
    torch.ones(4, 3),       # 4개 시점, 각 시점의 특성 3개
    torch.ones(2, 3) * 2,   # 2개 시점, 각 시점의 특성 3개
    torch.ones(5, 3) * 3    # 5개 시점, 각 시점의 특성 3개
]


# TODO B01: B가 첫 축인 [3, 5, 3] Padding Tensor를 만든다.

# batch_first의 기본값은 False이다.
# True로 설정하면 Batch가 첫 번째 차원으로 배치된다.

# BEGIN B01

# 길이가 서로 다른 시퀀스들을 하나의 텐서로 묶는다.
# 가장 긴 시퀀스의 길이인 5에 맞춰 부족한 부분을 0으로 채운다.
# batch_first=True:
# [Batch, Sequence Length, Feature] 순서로 텐서를 구성한다.
padded = pad_sequence(
    sequences,
    batch_first=True
)

# END B01


# 각 시퀀스의 원래 길이를 리스트 형태로 출력한다.
# len(x)는 각 시퀀스의 첫 번째 차원인 시점 개수를 반환한다.
print("입력 길이:", [len(x) for x in sequences])


# Padding이 적용된 최종 텐서의 형태를 출력한다.
# tuple()을 사용하면 텐서의 크기를 튜플 형태로 확인할 수 있다.
print("Padding shape:", tuple(padded.shape))


# 두 번째 시퀀스의 Padding 결과를 출력한다.
# 원래 길이는 2이지만 가장 긴 길이인 5에 맞춰
# 부족한 3개 시점이 0으로 채워져 있다.
print("두 번째 자료:\n", padded[1])