"""실습 04: 2 → 8 → 2 신경망의 순전파를 확인합니다."""
import torch
from torch import nn

torch.manual_seed(42)
# Sequential의 마지막 Linear를 수정해 출력 클래스 수가 2가 되게 하세요.
# 첫 Linear(2, 8)와 ReLU는 그대로 두세요. 모델 정의는 미리 제공했습니다.
# >>> B04
# TODO B04: 위 요구사항을 읽고 이 블록을 완성하세요.
# 신경망 모델을 순서대로 연결하여 생성합니다.
model = nn.Sequential(
    nn.Linear(2, 8),   # 입력 특성 2개를 받아서 뉴런 8개로 전달합니다.
    nn.ReLU(),        # 활성화 함수: 음수는 0으로 바꾸고 양수는 그대로 통과시킵니다.
    nn.Linear(8, 2)    # 뉴런 8개의 결과를 받아 최종 출력값 2개를 만듭니다.
)
# <<< B04

# 입력 데이터 4개를 생성합니다.
# 각 데이터는 특성 2개로 구성되어 있습니다.
X = torch.tensor([
    [1.0, 2.0],  # 첫 번째 데이터
    [3.0, 4.0],  # 두 번째 데이터
    [0.0, 1.0],  # 세 번째 데이터
    [2.0, 0.0]   # 네 번째 데이터
])

# 입력 데이터 X를 모델에 전달하여 예측 결과를 계산합니다.
# 입력 4개 각각에 대해 출력값 2개가 생성됩니다.
# 결과의 크기(shape)는 (4, 2)입니다.
logits = model(X)

print("입력:", tuple(X.shape))
print("출력:", tuple(logits.shape))
print("파라미터 수:", sum(p.numel() for p in model.parameters()))
# logits는 원시 점수입니다. 이 시점에는 학습하지 않았습니다.
