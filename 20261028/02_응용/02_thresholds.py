"""실습 02 응용: 가상 확률로 임계값만 바꿉니다. 최종 모델의 임계값을 탐색하는 코드가 아닙니다."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
# confusion_matrix: 실제 정답과 예측값을 비교하여 TN, FP, FN, TP 계산
# f1_score: 정밀도와 재현율을 종합한 F1 점수 계산
# roc_auc_score: 예측 확률을 이용해 AUC 점수 계산

# 현재 파이썬 파일이 있는 폴더를 기준으로 outputs 폴더 경로 생성
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# 실제 정답 데이터
# 0 = 성공, 1 = 실패라고 가정
y = np.array([0, 0, 1, 1])

score = np.array([0.10, 0.40, 0.35, 0.80])  # 각 데이터가 실패일 것이라고 예측한 가상 확률
rows = []   # 각 임계값에 따른 평가 결과를 저장할 빈 리스트

# 임계값을 0.5와 0.3으로 바꿔가며 평가 반복
for threshold in [0.5, 0.3]:
    # TODO A02: 확률이 threshold보다 클 때만 실패1로 바꾸세요. 같은 값은 성공0입니다.
    # y의 데이터 개수만큼 0으로 채워진 정수 배열 생성
    # len(y)는 4이므로 prediction은 [0, 0, 0, 0]으로 시작
    # 처음에는 모든 데이터를 성공(0)이라고 가정
    prediction = np.zeros(len(y), dtype=int)

     # score의 각 값이 threshold보다 큰지 비교
    # 예: score > 0.5의 결과는 [False, False, False, True]
    # 조건이 True인 위치만 선택하여 해당 prediction 값을 1로 변경
    # 즉, 임계값보다 실패 확률이 큰 데이터만 실패(1)로 예측
    # 임계값과 같은 값은 > 조건을 만족하지 않으므로 0으로 유지
    prediction[score > threshold] = 1
    # TODO 끝

    # 실제 정답 y와 예측값 prediction을 비교하여 혼동 행렬 계산
    # labels=[0, 1]: 클래스의 순서를 0, 1로 지정
    # ravel(): 혼동 행렬을 1차원으로 펼쳐 네 변수에 할당
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    # 현재 임계값과 평가 결과를 딕셔너리로 만들어 리스트에 추가
    rows.append({
        "threshold": threshold,         # 이번 반복에서 사용한 임계값
        "FP": fp,                       # 성공을 실패라고 잘못 예측한 건수
        "FN": fn,                       # 실제 실패를 성공이라고 놓친 건수
        "F1": f1_score(y, prediction),  # 현재 예측값의 F1 점수

        "assumed_cost": fp + 5*fn, # FP 비용은 1, FN 비용은 5라는 가정으로 총비용 계산

        # 실제 정답과 가상 확률을 이용한 AUC 점수
        # 임계값에 따라 달라지는 예측값이 아니라 score를 사용
        "AUC": roc_auc_score(y, score)
    })
# 평가 결과 리스트를 표(DataFrame)로 변환
result = pd.DataFrame(rows)
# 평가 결과를 소수점 셋째 자리까지 반올림하여 화면에 출력
# to_string(index=False): 행 번호를 생략
print(result.round(3).to_string(index=False))
# 평가 결과를 outputs 폴더 안의 CSV 파일로 저장
# index=False: 행 번호는 파일에 저장하지 않음
# encoding="utf-8-sig": 엑셀에서 한글이 깨지는 문제를 줄이기 위한 인코딩
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")

# 비용 계산에 사용한 가정을 안내
# 비용은 실제 화폐가 아닌 비교용 임의 점수
# 최종 평가에서는 임계값을 0.5로 고정한다는 점을 안내
print("cost는 FP 비용1, FN 비용5라는 가정입니다. 최종 평가 임계값은 사전에 0.5로 고정합니다.")
