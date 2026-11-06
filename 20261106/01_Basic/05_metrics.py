"""실습 05. 가상 기록 읽기. 이 표는 실제 모델 성능이 아닙니다."""
"""
[실습 05 - B06 기본 개념: 평가 기록과 최적 epoch 선택]

1. 가상 평가 기록
   - metrics_example.csv는 지표 읽기 연습을 위한 가상 기록
   - 이 코드에서는 모델 학습·추론·평가를 새로 실행하지 않음
   - CSV의 수치를 실제 YOLO 학습 성능으로 보고하면 안 됨

2. epoch와 검증 지표
   - epoch: 학습 데이터 전체를 한 번 사용하는 단위
   - 학습 중 epoch별 valid 평가 지표를 기록하면 모델 선택에 활용할 수 있음
   - 이 실습은 가상 기록의 map50_95가 가장 높은 epoch를 선택
   - 검증 지표로 선택한 결과는 독립적인 test 평가 결과와 구분해야 함

3. IoU와 AP
   - IoU: 예측 박스와 정답 박스의 교집합 면적 / 합집합 면적
   - 탐지 평가에서는 클래스와 박스 겹침을 함께 고려하여 예측을 판정
   - AP(Average Precision): 점수 기준을 변화시키며 얻는 Precision·Recall 관계를 요약
   - 클래스별 평가를 평균한 값이 mAP이며 단순한 정답 비율과는 다름

4. mAP50과 mAP50-95
   - mAP50: IoU 0.50을 정답 매칭 기준으로 사용한 AP를 클래스에 걸쳐 평균
   - mAP50-95: IoU 0.50~0.95를 0.05 간격으로 적용한 AP를 클래스·기준에 걸쳐 평균
   - 더 엄격한 위치 일치 기준까지 포함하므로 박스 위치의 정밀도도 확인하는 데 사용
   - 0~1로 기록된 값을 백분율로 표현하려면 100을 곱함
   - 이 코드는 AP를 계산하지 않고 이미 기록된 값을 읽음

5. mAP와 Accuracy의 차이
   - Accuracy는 일반적인 분류에서 전체 예측 중 정답 비율
   - 탐지는 물체 수와 위치, 클래스, 중복 예측, confidence 순서 등을 고려해야 함
   - 따라서 mAP=0.7을 이미지 70%를 맞혔다는 뜻으로 해석하면 안 됨

6. DataFrame과 열 선택
   - pd.read_csv(): CSV를 행과 열로 구성된 DataFrame으로 읽음
   - history["map50_95"]: 선택 기준인 한 열을 Series로 가져옴
   - history.idxmax()처럼 표 전체에 적용하면 각 열의 최대 위치가 반환되어 의도와 다름

7. idxmax()와 loc[]
   - Series.idxmax(): 최댓값 자체가 아니라 그 값이 있는 index 라벨을 반환
   - 최댓값이 여러 개면 첫 번째 항목의 index를 반환
   - history.loc[index]: 해당 index 라벨의 행 전체를 조회
   - 예: 최대 지표가 index 2에 있으면 loc[2]로 그 행을 선택
   - index와 epoch 번호는 서로 다른 값이므로 epoch는 선택한 행에서 따로 읽음

8. 결과 출력과 경로
   - best["epoch"]: 선택한 행의 epoch 값, int(): 정수로 변환하여 표시
   - best["map50"], best["map50_95"]: 같은 행의 지표 조회
   - Path.resolve().parent: 현재 소스 파일의 폴더
   - 이 코드는 선택 결과를 화면에 출력하며 별도 결과 파일을 저장하지 않음
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# CSV 평가 기록을 읽고 최적 행을 선택하는 표 처리 라이브러리
import pandas as pd

# 터미널 위치와 관계없이 현재 소스 폴더를 기준 경로로 사용
BASE = Path(__file__).resolve().parent

# 지표 읽기 연습용 가상 기록 CSV를 DataFrame으로 읽음
history = pd.read_csv(BASE / "data/metrics_example.csv")
# >>> B06
# 요구: map50_95가 가장 큰 행을 선택하세요. idxmax는 첫 최대값의 index를 반환합니다.
# TODO B06: 위 요구사항의 핵심 코드를 작성하세요.
# 1. history["map50_95"]로 선택 기준 열만 가져옴
# 2. idxmax()로 해당 열의 첫 최댓값이 위치한 index 라벨을 구함
# 3. loc[...]로 그 index의 행 전체를 선택하여 best에 저장
# best에는 해당 행의 epoch, map50, map50_95 등 열의 값이 함께 들어 있음
best = history.loc[history["map50_95"].idxmax()]
# <<< B06
# 수치의 출처를 명시하여 실제 모델 평가 결과와 구분
print("자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님")

# 선택한 행에서 epoch 값을 가져와 정수로 출력
print("선택 epoch:", int(best["epoch"]))

# 동일한 선택 행의 두 지표를 출력. 각각 다른 행의 최대값을 출력하는 것이 아님
print("mAP50:", best["map50"], "mAP50-95:", best["map50_95"])

# 지표의 의미와 평가 데이터의 역할을 안내
print("mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다.")
