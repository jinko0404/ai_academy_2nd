"""
[Day14 - A04 기본 개념: 클래스 가중치를 이용한 불균형 대응]

1. 클래스 불균형
   - 클래스마다 학습 이미지 수가 서로 다른 상태
   - 이 데이터의 train은 circle 120개, square 40개, triangle 20개로 구성됨
   - 다수 클래스만 잘 맞혀도 Accuracy가 높을 수 있어 소수 클래스 Recall도 확인

2. experiments와 비교 조건
   - (학습 전략, 불균형 대응 방식) 튜플을 리스트에 저장
   - ("head", "none"): Head만 학습하며 클래스 가중치를 사용하지 않음
   - ("head", "weight"): 같은 Head 학습에 클래스 가중치를 적용
   - 데이터, 초기값, epoch, 학습률을 고정하고 가중치 적용 여부만 비교

3. ImageFolder와 Tensor
   - ImageFolder: 폴더 이름으로 클래스와 숫자 라벨을 생성하는 데이터셋
   - dataset[i]: (이미지, 정답), dataset.targets: 정답 라벨 목록
   - Resize와 ToTensor: 이미지를 32×32, RGB float [0, 1]로 변환
   - torch.stack(): 이미지들을 (N, 3, 32, 32) 배치로 쌓음
   - torch.long: 정답 클래스 번호를 표현하는 정수 자료형
   - train과 valid의 클래스 이름·번호 매핑은 일치해야 함

4. Backbone과 Head
   - Backbone(0~8): 사전학습한 특징을 추출하며 두 실험에서 고정
   - Head(9): 256개 특징으로 3개 logit을 계산하며 두 실험에서 학습
   - source는 별도 합성 이미지로 학습한 소형 CNN이며 ImageNet ResNet18이 아님
   - logit: 확률로 변환하기 전 클래스별 점수

5. CNN과 nn.Sequential
   - Sequential: 등록된 층을 순서대로 실행
   - Conv2d: 특징 추출, BatchNorm2d: 채널별 정규화, ReLU: 비선형성 추가
   - MaxPool2d: 공간 크기 축소, AdaptiveAvgPool2d: 지정 크기로 평균 풀링
   - Flatten: (N, 16, 4, 4)를 (N, 256)으로 변환
   - Linear(256, 3): 256×3개 가중치와 3개 편향으로 총 771개 파라미터

6. 가중치 복원과 파라미터 동결
   - np.load(): .npz 배열 읽기, allow_pickle=False: pickle 객체 배열 로드 금지
   - state_dict(): 파라미터와 BatchNorm 통계 등 모델 상태를 이름별로 반환
   - torch.from_numpy(): 배열을 Tensor로 변환, load_state_dict(): 상태 복원
   - requires_grad=False: 해당 파라미터의 기울기를 계산하지 않음
   - eval(): BatchNorm 이동 통계를 고정하며 기울기 계산 자체를 끄지는 않음

7. 클래스 가중치 계산
   - torch.bincount(): 라벨별 개수 계산, minlength=3: 최소 3개 항목 확보
   - float(): 실수 Tensor로 변환, inverse=1/counts: 개수의 역수 계산
   - inverse / inverse.mean(): 가중치 평균이 1이 되도록 조정
   - [120, 40, 20]에서 가중치 [0.3, 0.9, 1.8]을 얻음
   - train의 개수만 사용하며 각 클래스가 train에 존재한다는 전제
   - 출력의 0.3000000119 등은 부동소수점 표현에 따른 차이

8. CrossEntropyLoss(weight=...)
   - logit과 정답 클래스 번호로 분류 손실을 계산하며 별도 Softmax가 필요 없음
   - weight: 정답 클래스에 따라 각 샘플 손실에 곱하는 클래스별 가중치
   - 기본 mean 감소에서는 가중 손실 합을 해당 샘플들의 가중치 합으로 나눔
   - weight=None: 클래스 가중치 없이 평균 손실 계산
   - 높은 가중치는 해당 클래스의 손실 기여도를 상대적으로 높임
   - 가중치를 적용한다고 모든 데이터에서 성능 향상이 보장되지는 않음

9. weight와 sampler의 차이
   - weight: 손실 계산에서 클래스별 중요도를 변경
   - sampler: 학습 샘플이 선택되는 확률이나 빈도를 변경
   - 이 파일은 sampler 없이 전체 train을 한 번에 사용하는 full-batch 학습 수행
   - 가중치는 train 손실에만 적용하고 valid 손실은 동일한 기준으로 비교

10. AdamW와 학습 순서
    - AdamW: 기울기를 이용하여 파라미터를 갱신하는 optimizer
    - lr=0.003: 학습률, weight_decay=0.0001: 가중치 감쇠 설정
    - zero_grad(): 기울기 초기화, backward(): 기울기 계산, step(): 갱신
    - range(1, 9): 1~8 epoch 학습

11. 최적 epoch 선택과 검증
    - no_grad(): 검증 시 기울기를 기록하지 않음
    - item(): 단일 원소 Tensor를 Python 숫자로 변환
    - valid 손실이 가장 낮은 epoch의 상태를 저장하고 최종 평가 전에 복원
    - detach().clone(): 미분 그래프에서 분리한 독립 복사본 생성
    - softmax(1): 클래스 확률 계산, argmax(1): 가장 높은 확률의 클래스 선택
    - valid로 모델을 선택하며 test는 읽지 않음

12. 평가 지표와 기록
    - Accuracy: 전체 정답 비율, Macro F1: 클래스별 F1의 동일 비중 평균
    - Recall: 실제 해당 클래스 중 올바르게 예측한 비율
    - classification_report(output_dict=True): 클래스별 지표를 딕셔너리로 반환
    - zero_division=0: 분모가 0인 지표를 0으로 처리
    - report['2']['recall']: 클래스 번호 2가 triangle이라는 매핑을 전제로 함
    - perf_counter(): 경과 시간 측정, to_csv(index=False): 행 번호 없이 CSV 저장
    - all_history는 epoch별 검증 손실을 메모리에 보관하며 파일로 저장하지 않음

13. 공통 실행 설정
    - Path: 경로 관리, resolve(): 절대 경로 변환, parent: 상위 폴더
    - mkdir(exist_ok=True): 폴더가 없으면 생성하고 이미 있으면 그대로 사용
    - set_num_threads(1): CPU 스레드 수 설정, manual_seed(14): 난수 시드 고정
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
# 파일 및 폴더 경로 관리
from pathlib import Path
# JSON 처리 도구이며 현재 파일에서는 사용하지 않음
import json
# source 가중치 배열 파일 읽기
import numpy as np
# 결과 표 작성과 CSV 저장
import pandas as pd
# Tensor 연산과 자동 미분
import torch
# 신경망 층과 손실 함수
from torch import nn
# 폴더 구조로 이미지와 정답 라벨 생성
from torchvision.datasets import ImageFolder
# 이미지 전처리 모듈
from torchvision import transforms as tr
# 분류 평가 함수들. confusion_matrix는 현재 파일에서 사용하지 않음
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# CPU 연산 스레드를 1개로 설정
torch.set_num_threads(1)
# 모델 생성 전에 시드를 다시 고정하여 실험별 초기값을 동일하게 설정
torch.manual_seed(14)
# 현재 파일의 폴더를 기준 경로로 사용
BASE = Path(__file__).resolve().parent
# 원본 데이터와 source 가중치 폴더
DATA = BASE / "data"
# 결과 CSV를 저장할 폴더
OUT = BASE / "outputs"
# outputs 폴더가 없으면 생성
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 소형 source 학습 당시 입력 규칙에 맞춰 크기와 픽셀 범위를 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
# 학습 구간의 경과 시간 측정
import time
# 학습용과 검증용 이미지에 동일한 전처리 적용
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)
# 각 항목의 이미지인 [0]을 가져와 전체 학습 배치로 쌓음
X_train = torch.stack([train[i][0] for i in range(len(train))])
# 학습 정답을 클래스 번호 정수 Tensor로 변환
y_train = torch.tensor(train.targets, dtype=torch.long)
# 검증 이미지를 데이터셋 순서대로 배치로 쌓음
X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
# 검증 이미지 순서에 대응하는 정답 Tensor 생성
y_valid = torch.tensor(valid.targets, dtype=torch.long)

# TODO A04: 같은 head 모델에서 none, weight 두 조건을 비교하세요.
# BEGIN A04
# Head 학습은 동일하게 유지하고 클래스 가중치 미적용·적용 두 조건을 비교
experiments = [("head", "none"), ("head", "weight")]
# END A04
# 실험별 최종 지표를 저장할 리스트
rows = []
# 모든 실험의 epoch별 검증 손실을 저장할 리스트
all_history = []
# 각 튜플에서 전략과 불균형 대응 방식을 꺼내 실험 실행
for strategy, balance in experiments:
    # 모델 생성 전에 시드를 다시 고정하여 실험별 초기값을 동일하게 설정
    torch.manual_seed(14)
    # 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
    # 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
    model = nn.Sequential(
        # [0~3] 3채널 → 8채널 특징 추출 후 공간 크기를 32×32에서 16×16으로 축소
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        # [4~6] 8채널 → 16채널 특징 추출. partial에서는 Conv(4)만 추가 학습
        nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
        # [7~9] 4×4 평균 풀링 → 256개 특징으로 펼침 → 클래스 점수 3개 출력
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 3),
    )
    # 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
    # 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
    with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
        # 현재 상태를 가져와 파일에 없는 항목은 초기값 유지
        state = model.state_dict()
        # 저장된 배열을 복사하고 Tensor로 변환하여 같은 이름의 상태를 덮어씀
        state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
    # source 상태를 실제 모델에 적용
    model.load_state_dict(state)

    # 전체 파라미터를 먼저 동결
    for p in model.parameters():
        p.requires_grad = False
    # 두 전략 모두 Head의 가중치와 편향 학습 허용
    for p in model[9].parameters():
        p.requires_grad = True
    # partial 전략에서만 두 번째 Conv의 가중치 학습 허용
    if strategy == "partial":
        for p in model[4].parameters():
            p.requires_grad = True
    # 학습 중에도 BatchNorm의 이동 통계를 고정하며 기울기 계산은 유지
    model.eval()
    # 학습 가능한 파라미터만 AdamW에 등록
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003, weight_decay=0.0001)
    # train의 클래스별 이미지 개수를 실수 Tensor로 계산
    counts = torch.bincount(y_train, minlength=3).float()
    # 적은 클래스에 더 큰 값이 부여되도록 개수의 역수 계산
    inverse = 1.0 / counts
    # 클래스 가중치의 평균을 1로 조정
    class_weights = inverse / inverse.mean()
    # weight 조건일 때만 가중치를 사용함. 현재 두 실험은 모두 일반 교차 엔트로피 사용
    criterion = nn.CrossEntropyLoss(weight=class_weights if balance == "weight" else None)
    # 첫 검증 손실을 최적값으로 기록할 수 있도록 양의 무한대로 초기화
    best_loss = float("inf")
    # 학습과 epoch별 검증을 포함하는 구간의 시작 시간 기록
    started = time.perf_counter()
    # 1~8 epoch 학습
    for epoch in range(1, 9):
        # 이전 기울기 초기화
        optimizer.zero_grad()
        # 전체 train 배치의 logit과 정답으로 손실 계산
        loss = criterion(model(X_train), y_train)
        # 학습 가능한 파라미터의 기울기 계산
        loss.backward()
        # 파라미터 갱신
        optimizer.step()
        # 복원한 모델의 검증 예측 수행
        with torch.no_grad():
            # train 가중치 적용 여부와 관계없이 일반 교차 엔트로피로 valid를 평가
            vl = nn.CrossEntropyLoss()(model(X_valid), y_valid).item()
        # 실험 설정과 현재 epoch의 검증 손실을 기록
        all_history.append({"strategy":strategy,"balance":balance,"epoch":epoch,"valid_loss":vl})
        # 이전 최적값보다 검증 손실이 낮을 때만 최적 모델 갱신
        if vl < best_loss:
            # 가장 낮은 검증 손실과 해당 epoch 번호 보관
            best_loss=vl;best_epoch=epoch
            # 이후 학습으로 값이 바뀌지 않도록 모델 상태를 분리하여 복사
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    # 학습 구간의 경과 시간 계산. 아래 최적 상태 복원과 최종 평가 시간은 제외
    elapsed = time.perf_counter()-started
    # 검증 손실이 가장 낮았던 epoch의 상태 복원
    model.load_state_dict(best_state)
    # 복원한 모델의 검증 예측 수행
    with torch.no_grad():
        # 클래스별 확률로 변환하고 각 이미지의 예측 클래스 번호 선택
        prob=model(X_valid).softmax(1);pred=prob.argmax(1)
    # 클래스별 지표를 딕셔너리로 생성하며 분모가 0인 지표는 0으로 처리
    report=classification_report(y_valid,pred,labels=[0,1,2],output_dict=True,zero_division=0)
    # 전략, 학습 원소 수, 최적 epoch, 정확도, macro F1, triangle Recall, 시간을 기록
    # report['2']는 현재 데이터의 클래스 번호 2인 triangle에 해당
    rows.append({"strategy":strategy,"balance":balance,"params":sum(p.numel() for p in model.parameters() if p.requires_grad),"best_epoch":best_epoch,"accuracy":accuracy_score(y_valid,pred),"macro_f1":f1_score(y_valid,pred,labels=[0,1,2],average="macro",zero_division=0),"triangle_recall":report['2']['recall'],"seconds":elapsed})

# Tensor를 Python 리스트로 변환하여 train에서 계산한 가중치 출력
# 예상: 약 [0.3, 0.9, 1.8]이며 부동소수점 표현 때문에 긴 소수가 나타날 수 있음
print("train class_weights:",class_weights.tolist())
# 실험별 결과를 표로 만들고 시간 열을 포함하여 CSV로 저장
table=pd.DataFrame(rows);table.to_csv(OUT/"weight_comparison.csv",index=False)
# 화면에서는 시간 열을 제외하고 소수점 넷째 자리까지 반올림하여 출력
print(table.drop(columns="seconds").round(4).to_string(index=False))


"""
결론
    이번 데이터와 설정에서는 클래스 가중치 적용으로 검증 성능이 개선됨.
    다만 valid는 최적 epoch 선택에도 사용했으므로 독립적인 최종 평가 결과는 아님.
    이 결과만으로 새로운 이미지에서도 항상 100%이거나 weight가 항상 유리하다고 단정할 수 없음.
    test는 사용하지 않았으며 이번 수치는 valid에 대한 결과로 기록함.
"""