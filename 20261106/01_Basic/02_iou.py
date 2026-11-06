"""실습 02. 손으로 정한 예측 4개를 정답 3개와 비교합니다. YOLO 결과가 아닙니다."""
"""
[실습 02 - B02·B03 기본 개념: IoU와 예측·정답 매칭]

1. 예측 박스와 GT
   - pred: 손으로 만든 예측 예시, truth: 정답 박스(Ground Truth)
   - 이 실습의 예측은 YOLO 모델을 실행한 결과가 아님
   - class_id: 클래스 번호, confidence: 예측 예시에 지정된 점수
   - xyxy는 왼쪽 위 (x1, y1)와 오른쪽 아래 (x2, y2) 좌표

2. 교집합 면적(intersection)
   - 두 박스가 함께 차지하는 영역의 면적
   - 겹치는 왼쪽 경계는 x1 중 큰 값, 오른쪽 경계는 x2 중 작은 값
   - 겹치는 위 경계는 y1 중 큰 값, 아래 경계는 y2 중 작은 값
   - 오른쪽-왼쪽, 아래-위가 음수이면 겹치지 않으므로 max(0, ...)으로 0 처리
   - 교집합 면적 = 겹치는 폭 × 겹치는 높이

3. 합집합 면적(union)
   - 두 박스 중 하나 이상에 포함되는 전체 영역의 면적
   - 합집합 = 예측 박스 면적 + 정답 박스 면적 - 교집합 면적
   - 교집합은 두 박스 면적에 중복 포함되므로 한 번 빼야 함
   - 이 실습은 폭=x2-x1, 높이=y2-y1 규칙을 사용하며 +1을 더하지 않음

4. IoU(Intersection over Union)
   - IoU = 교집합 면적 / 합집합 면적
   - 정상적인 양의 면적 박스에서는 0~1 범위
   - 겹치지 않으면 0, 완전히 일치하면 1
   - 위치의 겹침만 측정하므로 IoU가 높아도 클래스가 다르면 올바른 탐지가 아님
   - 제공 데이터의 union은 0이 아니므로 별도 0 나눗셈 처리를 하지 않음

5. 일대일 매칭과 used
   - TP: 같은 클래스의 정답과 기준 IoU 이상으로 연결된 예측
   - FP: 정답과 연결되지 못한 예측, FN: 예측과 연결되지 못한 정답
   - 하나의 정답을 여러 예측의 TP로 중복 계산하지 않도록 used에 기록
   - set(): 중복 없는 원소를 보관하며 j in used로 사용 여부 검사
   - 이 실습의 매칭 기준은 IoU >= 0.5

6. 예측 처리 순서와 IoU 정렬
   - 예측 CSV가 confidence 내림차순이라는 전제로 행 순서대로 처리
   - 코드가 pred를 confidence 기준으로 직접 정렬하는 것은 아님
   - np.asarray(): IoU 목록을 배열로 변환
   - np.argsort(-배열): 원래 IoU가 큰 순서대로 인덱스를 반환
   - 클래스가 다르거나 이미 연결된 정답은 continue로 건너뜀
   - B03은 매칭 여부를 결정하는 미완성 영역이며 현재 pass는 아무 작업도 하지 않음

7. Precision과 Recall
   - Precision = TP / 전체 예측 수: 예측 중 올바르게 연결된 비율
   - Recall = TP / 전체 정답 수: 정답 중 예측으로 찾은 비율
   - 이 코드는 예측·정답이 존재하는 정상 입력을 전제로 계산
   - 하나의 IoU 기준에서 계산한 결과이며 AP나 mAP를 계산하는 코드는 아님

8. 표의 max_iou와 주의점
   - max_iou: 해당 예측과 모든 정답의 IoU 중 최댓값
   - 클래스나 정답 사용 여부를 고려하지 않은 값이므로 실제 매칭 IoU와 다를 수 있음
   - matched가 True이면 TP, 아니면 FP로 표시
   - B03이 미완성이면 매칭이 기록되지 않아 지표가 올바르게 완성되지 않음

9. pandas와 결과 저장
   - read_csv(): CSV 읽기, itertuples(): 각 행을 튜플 형태로 순회
   - iloc[j]: 표의 j번째 위치 행 조회
   - DataFrame(): 결과를 표로 구성, to_string(index=False): 행 번호 없이 출력
   - to_csv(index=False): 행 번호 없이 저장
   - utf-8-sig: UTF-8에 BOM을 붙여 일부 프로그램의 한글 인식에 도움
   - Path.resolve().parent: 현재 소스 폴더, round(..., 4): 소수점 넷째 자리 반올림
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# IoU 목록의 배열 변환과 정렬 인덱스 계산
import numpy as np
# 예측·정답 CSV 읽기와 결과 표 저장
import pandas as pd
# 터미널 위치와 관계없이 현재 소스 폴더를 기준 경로로 사용
BASE = Path(__file__).resolve().parent
# confidence가 높은 순서로 준비된 예측 예시 CSV를 읽음
pred = pd.read_csv(BASE / "data/boxes_fixture.csv")
# 비교할 정답 박스 CSV를 읽음
truth = pd.read_csv(BASE / "data/boxes_gt.csv")
# 이미 연결된 정답의 행 위치를 보관하여 중복 TP 방지
used = set()
# 예측별 클래스, confidence, 최대 IoU, 판정 결과를 모을 리스트
rows = []
# 예측을 CSV 순서대로 하나씩 처리
for p in pred.itertuples():
    # 현재 예측과 모든 정답 사이의 IoU를 정답 행 순서대로 저장
    overlaps = []
    # 현재 예측을 정답 박스 각각과 비교
    for g in truth.itertuples():
        # 겹치는 x 구간 길이를 계산하고 겹치지 않으면 0 처리
        width = max(0, min(p.x2, g.x2) - max(p.x1, g.x1))
        # 겹치는 y 구간 길이를 계산하고 겹치지 않으면 0 처리
        height = max(0, min(p.y2, g.y2) - max(p.y1, g.y1))
        # 겹치는 폭과 높이를 곱해 교집합 면적 계산
        intersection = width * height
        # 두 박스 면적을 더하고 중복 계산된 교집합을 한 번 뺌
        union = (p.x2-p.x1)*(p.y2-p.y1) + (g.x2-g.x1)*(g.y2-g.y1) - intersection
        # >>> B02
        # 요구: intersection을 union으로 나누어 iou를 계산하세요.
        # 이 실습의 데이터는 모두 양의 면적이므로 union은 0이 아닙니다.
        # TODO B02: 위 요구사항의 핵심 코드를 작성하세요.
        # 전체 합집합 중 교집합이 차지하는 비율 계산. 이 데이터에서는 union > 0
        iou = intersection / union
        # <<< B02
        # 현재 정답과의 IoU를 목록 끝에 추가
        overlaps.append(iou)
    # 현재 예측은 아직 정답과 연결되지 않은 상태로 시작
    matched = False
    # 현재 예측과 IoU가 높은 정답부터 차례로 매칭 후보 검사
    for j in np.argsort(-np.asarray(overlaps)):
        # 클래스가 다르거나 이미 다른 예측에 연결된 정답이면 제외
        if truth.iloc[j].class_id != p.class_id or j in used:
            continue
        # >>> B03
        # 요구: 같은 클래스의 미사용 정답에 IoU>=0.5이면 연결하고 used에 기록하세요.
        pass  # TODO B03: 위 요구사항의 핵심 코드를 작성하세요.
        # <<< B03
    # 전체 정답 중 최대 IoU와 현재 매칭 판정을 기록
    rows.append([p.class_id, p.confidence, round(max(overlaps), 4), "TP" if matched else "FP"])
# 예측별 결과를 이름 있는 열로 구성
table = pd.DataFrame(rows, columns=["class", "confidence", "max_iou", "result"])
# 행 번호 없이 결과 표 출력
print(table.to_string(index=False))
# 일대일 매칭된 정답 수가 TP. 남은 예측은 FP, 남은 정답은 FN
tp = len(used); fp = len(pred)-tp; fn = len(truth)-tp
# 매칭 결과 개수 출력
print("TP/FP/FN:", tp, fp, fn)
# 정밀도와 재현율을 계산하고 소수점 넷째 자리까지 반올림
print("Precision/Recall:", round(tp/len(pred), 4), round(tp/len(truth), 4))
# 현재 소스 폴더에 예측별 결과 CSV 저장
table.to_csv(BASE / "p02_matching.csv", index=False, encoding="utf-8-sig")
