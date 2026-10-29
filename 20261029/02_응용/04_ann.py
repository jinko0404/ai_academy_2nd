"""실습 04: 은닉 너비에 따라 파라미터 수가 변합니다. 학습 성능 비교는 아닙니다."""
import torch
from torch import nn

torch.manual_seed(42)
X = torch.ones(4, 2)
for hidden in [4, 8, 16]:
    # hidden 값을 사용해 입력 2개 → 은닉 hidden개 → 출력 2개를 연결하세요.
    # 은닉층 뒤에 ReLU를 넣고 마지막에 softmax는 넣지 않습니다.
    # >>> A04
    # TODO A04: 위 요구사항을 읽고 이 블록을 완성하세요.
    # 입력 2개 → 은닉층 hidden개 → 출력 2개
    # 은닉층은 입력 데이터와 최종 출력 사이에서 중간 계산을 수행하는 층
    model = nn.Sequential(
        nn.Linear(2, hidden),  # 입력 2개를 받아 은닉층 hidden개로 전달
        nn.ReLU(),             # 은닉층의 출력에 활성화 함수 적용
        nn.Linear(hidden, 2)   # 은닉층의 결과를 출력 2개로 변환
    )
    # <<< A04
    logits = model(X)
    count = sum(p.numel() for p in model.parameters())
    print("은닉:", hidden, "출력:", tuple(logits.shape), "파라미터:", count)
