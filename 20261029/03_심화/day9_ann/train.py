"""학습 한 epoch와 평가를 분리합니다. 빈 데이터·불량 자료는 이번 단계의 과제가 아닙니다."""
import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader: #학습 데이터를 batch 단위로 가져온다.
        xb, yb = xb.to(device), yb.to(device)   # 가져온 batch를 GPU 혹은 CPU로 이동
        optimizer.zero_grad()                   # 이전 학습에서 남아있는 기울기를 초기화시킨다.
        loss = loss_fn(model(xb), yb)           # 순전파를 진행해 예측 점수를 계산하고 손실을 구한다.
        # loss.backward() 후 optimizer.step()을 호출하세요.
        # 매 batch마다 초기화하고 새로운 gradient로 갱신해야 합니다.
        # >>> H07
        # TODO H07: 위 요구사항을 읽고 이 블록을 완성하세요.
        loss.backward()     # 역전파 진행하여 손실에 대한 기울기 계산
        optimizer.step()    # 계산된 기울기를 토대로 모델의 가중치와 편향을 갱신
        # <<< H07
        total += loss.item() * len(yb)
        count += len(yb)
    # 마지막 작은 batch도 실제 샘플 수만큼 반영합니다.
    return total / count


def evaluate(model, X, y, loss_fn, device):
    model.eval()
    with torch.no_grad():
        #모델에 입력 데이터 X를 넣어 예측 점수를 계산하였다.
        logits = model(X.to(device)) # 모델의 예측 점수

        #예측점수(logits)와 실제 정답(y)를 비교하여 손실을 계산한다.
        #.item()은 손실 값을 파이썬 숫자로 변환하는 함수
        loss = loss_fn(logits, y.to(device)).item()
        # logits에서 클래스 인덱스를 구하고 CPU로 옮겨 prediction에 저장하세요.
        # 평가에서는 backward와 optimizer.step을 호출하지 않습니다.
        # >>> H08
        # TODO H08: 위 요구사항을 읽고 이 블록을 완성하세요.
        # 각 데이터의 예측 점수 중 가장 높은 점수의 클래스 인덱스를 선택
        # dim=1은 각 데이터의 클래스별 점수를 비교한다는 뜻
        # .cpu()는 예측 결과를 CPU로 이동
        prediction = logits.argmax(dim=1).cpu()
        # <<< H08
    return loss, (prediction == y.cpu()).float().mean().item(), prediction
