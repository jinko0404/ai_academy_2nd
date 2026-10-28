"""실습 02 심화: ROC 좌표와 두 임계값 표를 별도로 기록합니다."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

y = np.array([0, 0, 1, 1])                  # 실제 정답 데이터
score = np.array([0.10, 0.40, 0.35, 0.80])  # 각 데이터가 양성(실패)일 것이라고 예측한 가상 확률

# 여러 임계값에서 ROC 곡선을 구성하는 좌표를 계산
# fpr: 실제 음성 중 양성으로 잘못 예측한 비율(FP 비율)
# tpr: 실제 양성 중 양성으로 올바르게 예측한 비율(TP 비율, 재현율)
# thresholds: 각 ROC 좌표에 대응하는 임계값
fpr, tpr, thresholds = roc_curve(y, score)

# ROC 좌표와 해당 임계값을 하나의 표로 정리해 CSV 파일로 저장
# 각 배열의 값들이 같은 행에서 서로 대응함
pd.DataFrame({
    "FPR": fpr, 
    "TPR": tpr, 
    "threshold": thresholds
}).to_csv(OUT / "roc_points.csv", index=False, encoding="utf-8-sig")

rows = []# 두 임계값의 평가 결과를 저장할 빈 리스트

#임계값 0.5 0.3으로 바꿔가며 반복
for threshold in [0.5, 0.3]:
    # TODO H02: 현재 임계값보다 점수가 클 때만 양성1로 바꾸세요.
    # 데이터 개수만큼 0으로 채워진 정수 배열 생성
    # 모든 데이터를 우선 음성(0)이라고 가정
    pred = np.zeros(len(y), dtype=int)

    # 각 데이터의 점수가 임계값보다 큰 위치만 선택
    # 선택된 위치의 예측값을 양성(1)으로 변경
    # 임계값과 점수가 같으면 0으로 유지
    pred[score > threshold] = 1
    # TODO 끝

    # 실제 정답과 예측값을 비교해 혼동 행렬 계산
    # labels=[0, 1]: 클래스의 순서를 0, 1로 지정
    # ravel(): 혼동 행렬을 1차원으로 펼쳐 네 변수에 할당
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # 현재 임계값의 평가 결과를 딕셔너리로 만들어 리스트에 추가
    rows.append({
        "threshold": threshold,     # 현재 적용한 임계값
        "FP": fp,                   # 음성을 양성이라고 잘못 예측한 건수
        "FN": fn,                   # 실제 양성을 음성이라고 놓친 건수
        "F1": f1_score(y, pred),    # 현재 예측 결과의 F1 점수

        "cost": fp + 5*fn,  # FP 비용은 1, FN 비용은 5라는 가정으로 총비용 계산

        # 실제 정답과 예측 확률을 이용해 AUC 계산
        # 현재 반복문의 threshold 값은 AUC 계산에 직접 사용되지 않음
        "AUC": roc_auc_score(y, score)
    })
result = pd.DataFrame(rows)
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("ROC 첫 임계값 inf는 어떤 값도 양성으로 정하지 않는 시작점입니다. 입력 오류가 아닙니다.")
print("이 원리 예제는 최종 분류기의 임계값을 변경하는 실험이 아닙니다.")
