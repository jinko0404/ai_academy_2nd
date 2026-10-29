"""실습 06: 작은 정상 자료로 학습 순서만 익힙니다. 여기는 train 오차 확인용입니다."""
import torch
from torch import nn

torch.manual_seed(42)
X = torch.tensor([[-2., -1.], [-1., -1.], [-1., 0.], [0., -1.],
                  [1., 0.], [0., 1.], [1., 1.], [2., 1.]])
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print("학습 전 loss:", round(loss_fn(model(X), y).item(), 6))
model.train()
for epoch in range(80):
    # 1.zero_grad() : 이전 학습에서 남아있던 기울기 초기화
    optimizer.zero_grad()

    # 2.순전파 진행 및 모델의 예측 결과와 실제 정답을 비교해서 손실(loss)을 계산
    loss = loss_fn(model(X), y)
    # 현재 loss의 gradient를 계산하세요.
    # 파라미터 값은 아직 바뀌지 않으며 .grad에 기울기가 들어갑니다.
    # >>> B06
    # TODO B06: 위 요구사항을 읽고 이 블록을 완성하세요.
    # 3.역전파 수행하여 로스율의 기울기를 구한다.
    loss.backward()
    # <<< B06
    # 현재 gradient를 사용해 파라미터를 한 번 갱신하세요.
    # optimizer의 step 메서드를 호출합니다.
    # >>> B07
    # TODO B07: 위 요구사항을 읽고 이 블록을 완성하세요.
    # 4.계산된 기울기를 이용해 모델의 가중치와 편향을 갱신한다. 
    # 가중치 수정
    optimizer.step()
    # <<< B07
model.eval()
with torch.no_grad():
    logits = model(X)
    print("학습 후 loss:", round(loss_fn(logits, y).item(), 6))
    print("train 정확도:", round((logits.argmax(dim=1) == y).float().mean().item(), 4))
print("주의: 위 정확도는 학습한 8건의 결과이며 새로운 자료의 성능이 아닙니다.")
