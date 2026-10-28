"""실습 01 응용: 모델 예시와 전부 성공으로 예측하는 기준선을 비교합니다."""
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# confusion_matrix: 실제 정답과 예측값을 비교하여 TN, FP, FN, TP 계산
# accuracy_score: 정확도 계산
# precision_score: 정밀도 계산
# recall_score: 재현율 계산
# f1_score: 정밀도와 재현율을 종합한 F1 점수 계산

# 실제 정답 데이터
# 0이 14개, 1이 6개
# 여기서는 0 = 성공, 1 = 실패라고 가정
y = [0]*14 + [1]*6

# 비교할 두 가지 예측 결과를 딕셔너리로 저장
predictions = {
    "example": [0]*10 + [1]*4 + [0]*4 + [1]*2, # example: 성공 10건, 실패 4건, 성공 4건, 실패 2건으로 예측
    "all_success": [0]*20 # all_success: 전체 20건을 모두 성공(0)이라고 예측
    }
rows = []# 각 모델의 평가 결과를 저장할 빈 리스트

# 딕셔너리(15~19라인 참고)에서 모델 이름(name)과 예측 결과(pred)를 하나씩 가져옴
for name, pred in predictions.items():
    # 실제 정답 y와 예측값 pred를 비교해 혼동 행렬을 계산
    # labels=[0, 1]: 0과 1의 순서를 지정
    # ravel(): 2차원 혼동 행렬을 1차원으로 펼쳐 네 변수에 할당
    # tn = 실제 0을 0으로 예측한 수
    # fp = 실제 0을 1로 잘못 예측한 수
    # fn = 실제 1을 0으로 잘못 예측한 수
    # tp = 실제 1을 1로 올바르게 예측한 수
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # TODO A01: 현재 후보의 F1을 계산하세요. 분모 0은 0으로 표시합니다.
    # F1 점수 손계산: f1 = (2 * tp) / ((2 * tp) + fp + fn)
    # y: 실제 정답
    # pred: 모델의 예측 결과
    # zero_division=0: 정밀도나 재현율 계산에서 분모가 0이면 0으로 처리
    f1 = f1_score(y, pred, zero_division=0) # 함수 설명(8라인 참고)
    # TODO 끝
    # 현재 모델의 평가 결과를 딕셔너리로 만들어 리스트에 추가
    rows.append({"model": name, "Accuracy": accuracy_score(y, pred),
                 "Precision": precision_score(y, pred, zero_division=0),
                 "Recall": recall_score(y, pred, zero_division=0), "F1": f1,
                 "FP": fp, "FN": fn, "TP": tp})

# 위 반복문에서 완성된 평가 결과 리스트를 표(DataFrame)형태로 변환하여 출력
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("정확도가 높은 all_success도 실패 6건을 하나도 찾지 못합니다.")
