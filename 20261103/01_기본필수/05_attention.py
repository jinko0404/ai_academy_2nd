import torch
from torch import nn

"""
============================================================
[개념 정리] Multi-Head Attention (멀티헤드 어텐션)
============================================================

1. Attention (어텐션)
   - 입력 데이터에서 서로 어떤 관계가 있는지 계산하는 방법
   - 각 데이터가 다른 데이터에 얼마나 집중해야 하는지 가중치를 계산한다.

2. Query (쿼리)
   - 현재 내가 어떤 정보를 찾고 있는지를 나타내는 값

3. Key (키)
   - 다른 데이터가 어떤 정보를 가지고 있는지 나타내는 값
   - Query와 비교하여 얼마나 관련 있는지 계산한다.

4. Value (밸류)
   - 실제로 가져올 정보
   - Attention 가중치를 적용하여 정보를 조합한다.

5. Self-Attention (셀프 어텐션)
   - Query, Key, Value에 동일한 입력 데이터를 사용하는 방식
   - 입력 데이터 내부의 관계를 계산한다.

6. Multi-Head Attention (멀티헤드 어텐션)
   - 여러 개의 Attention을 병렬로 실행하는 방식
   - 각 Head가 서로 다른 관점에서 데이터의 관계를 학습한다.

7. Head (헤드)
   - 독립적으로 Attention 계산을 수행하는 단위
   - 여기서는 4개의 Head를 사용한다.

8. Attention Weight (어텐션 가중치)
   - 각 데이터가 다른 데이터에 얼마나 집중하는지를 나타내는 값
   - 각 Query에 대한 가중치의 합은 1이다.

9. average_attn_weights
   - True: 모든 Head의 Attention Weight를 평균으로 반환
   - False: Head별 Attention Weight를 각각 반환

10. batch_first=True
    - 입력 데이터의 차원을 (Batch, Sequence, Feature) 순서로 사용한다.

"""

torch.set_num_threads(1)
# CPU 연산에 사용할 스레드 수를 1개로 고정한다.

torch.manual_seed(42)
# 난수 시드를 고정하여 실행할 때마다 동일한 난수가 생성되도록 한다.

# 1. 입력 데이터 생성
x = torch.randn(2, 5, 16)
# 입력 데이터 크기: (Batch, Sequence, Feature)
# 2: 데이터 묶음 개수
# 5: 데이터 묶음 하나에 포함된 데이터 개수
# 16: 데이터 하나의 특징 차원

# 2. Multi-Head Attention 생성
attention = nn.MultiheadAttention(
    embed_dim=16,
    num_heads=4,
    dropout=0.0,
    batch_first=True
)
# embed_dim=16: 입력 특징 차원
# num_heads=4: Attention Head 4개 사용
# Head 하나당 특징 차원은 16 / 4 = 4
# dropout=0.0: Dropout 적용 안 함
# batch_first=True: Batch를 첫 번째 차원으로 사용

# 3. Self-Attention 실행
output, weights = attention(
    x, x, x,                     # Query, Key, Value에 동일한 x를 전달하므로 Self-Attention이다.
    average_attn_weights=False   # 4개 Head의 Attention Weight를 평균내지 않고 각각 반환한다.
)
# output: Attention 계산 후 생성된 출력 데이터
# weights: 각 Head가 계산한 Attention 가중치

# 4. 결과 확인
print("입력:", tuple(x.shape))
# 결과: (2, 5, 16)

print("출력:", tuple(output.shape))
# 결과: (2, 5, 16)

print("head별 weight:", tuple(weights.shape))
# 결과: (2, 4, 5, 5)
# 2: Batch / 4: Head / 5: Query / 5: Key

print("각 행의 합:", weights[0, 0].sum(dim=-1).detach().round(decimals=4))
# 첫 번째 Batch의 첫 번째 Head를 선택한다.
# 마지막 차원(Key)의 가중치를 모두 더한다.
# 각 Query의 가중치 합이 1인지 확인한다.
