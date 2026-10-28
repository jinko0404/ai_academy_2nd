"""실습 01 심화: 지표와 업무 비용을 함께 읽습니다. 기존 응용 수준의 짧은 비교입니다."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# confusion_matrix: 실제 정답과 예측값을 비교해 TN, FP, FN, TP 계산
# accuracy_score: 정확도 계산
# precision_score: 정밀도 계산
# recall_score: 재현율 계산
# f1_score: 정밀도와 재현율을 종합한 F1 점수 계산

# 현재 파이썬 파일이 있는 폴더를 기준으로 outputs 폴더 경로 생성
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# 실제 정답 데이터
# 0이 14개, 1이 6개
# 0 = 성공, 1 = 실패라고 가정
y = [0]*14 + [1]*6

# 비교할 두 가지 모델의 예측 결과를 딕셔너리에 저장
predictions = {
    "example": [0]*10 + [1]*4 + [0]*4 + [1]*2, # example: 성공 10건, 실패 4건, 성공 4건, 실패 2건으로 예측
    "all_success": [0]*20 # all_success: 전체 20건을 모두 성공(0)이라고 예측
}
rows = [] #각 모델 평가 결과를 저장할 리스트

for name, pred in predictions.items():
    # 실제 정답 y와 예측값 pred를 비교해 혼동 행렬 계산
    # labels=[0, 1]: 정답 클래스의 순서를 0, 1로 지정
    # ravel(): 2차원 혼동 행렬을 1차원으로 펼침
    # 펼쳐진 값을 tn, fp, fn, tp 순서대로 할당
    # tn = 실제 0을 0으로 예측한 건수
    # fp = 실제 0을 1로 잘못 예측한 건수
    # fn = 실제 1을 0으로 잘못 예측한 건수
    # tp = 실제 1을 1로 올바르게 예측한 건수
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # TODO H01: FP 비용1, FN 비용5라는 가정으로 총비용을 계산하세요.
    # fp * 1: 성공을 실패라고 잘못 예측한 건수에 비용 1을 곱함
    # fn * 5: 실제 실패를 성공이라고 놓친 건수에 비용 5를 곱함
    # 두 비용을 더해 총비용을 계산
    cost = (fp * 1) + (fn * 5)
    # TODO 끝

    # 현재 모델의 평가 결과를 딕셔너리로 만들어 리스트에 추가
    rows.append({
        "model": name,  # 모델 이름
        
        "Accuracy": accuracy_score(y, pred),                    # 전체 예측 중 정답 비율
        "Precision": precision_score(y, pred, zero_division=0), # 실패라고 예측한 것 중 실제 실패 비율
        "Recall": recall_score(y, pred, zero_division=0),       # 실제 실패 중 찾아낸 비율
        "F1": f1_score(y, pred, zero_division=0),               # 정밀도와 재현율을 종합한 점수
        
        "FP": fp,   # 성공을 실패라고 잘못 예측한 건수
        "FN": fn,   # 실제 실패를 성공이라고 놓친 건수
        
        "assumed_cost": cost    # 가정한 비용 기준으로 계산한 총비용
    })
result = pd.DataFrame(rows)
result.to_csv(OUT / "metric_comparison.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("가정: 오탐 1회 비용1, 실제 실패를 놓친 비용5. 비용 단위는 임의 점수입니다.")
