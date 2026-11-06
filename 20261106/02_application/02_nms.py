"""실습 02. 고전 NMS 원리 실험. YOLO26 기본 추론에 NMS가 있다고 가정하지 않습니다."""
"""
[실습 02 - A02 기본 개념: 클래스별 NMS와 탐지 지표]

1. 예측 예시와 정답
   - boxes_fixture.csv는 손으로 준비한 예측 예시이며 실제 모델 추론 결과가 아님
   - boxes_gt.csv는 정답 박스(GT)
   - 이 코드는 고전 NMS 원리를 실험하며 특정 모델의 기본 추론 동작을 검증하지 않음

2. Tensor와 confidence 필터
   - boxes: (N, 4) xyxy 좌표, scores: 예측 점수, classes: 클래스 번호
   - 좌표·점수는 float32, 클래스 번호는 int64
   - torch.where(scores >= conf)[0]: 기준을 통과한 원본 예측 인덱스 반환
   - confidence 기준을 높이면 후보 수는 줄거나 같지만 성능 향상이 보장되지는 않음

3. NMS(Non-Maximum Suppression)
   - 점수가 높은 박스를 우선 남기고 과도하게 겹치는 낮은 점수 박스를 억제
   - 정답 없이 예측 박스들 사이의 중복을 줄이는 후처리
   - batched_nms는 같은 클래스끼리만 억제하며 다른 클래스끼리는 억제하지 않음
   - 인자: 박스 좌표, 점수, 클래스 번호, IoU 임계값
   - 기준보다 IoU가 큰 같은 클래스의 낮은 점수 박스를 억제
   - NMS IoU 임계값이 높을수록 중복 제거가 느슨해짐
   - 반환값은 남길 박스의 위치이며 점수 내림차순으로 정렬됨

4. A02의 인덱스 변환
   - ids: confidence 기준을 통과한 원본 위치
   - selected: ids로 줄인 배열 안에서 NMS가 남긴 위치
   - keep = ids[selected]: 다시 원본 배열의 위치로 변환
   - 예: ids=[0,2,3], selected=[0,2]이면 keep=[0,3]
   - selected를 원본 boxes에 바로 사용하면 다른 박스를 선택할 수 있음

5. IoU와 정답 매칭
   - IoU = 교집합 면적 / 합집합 면적. 완전 일치는 1, 겹치지 않으면 0
   - box_iou(): 두 박스 집합 사이의 IoU 행렬 반환
   - 같은 클래스의 미사용 정답 중 IoU가 높은 순서로 연결 시도
   - 정답 매칭은 IoU >= 0.5로 고정하며 NMS용 임계값과 별개
   - used 집합으로 하나의 정답을 여러 번 TP로 계산하지 않도록 처리

6. TP, FP, FN과 지표
   - TP: 정답과 연결된 예측, FP: 정답과 연결되지 못한 남은 예측
   - FN: 예측과 연결되지 못한 정답
   - Precision = TP / 남긴 예측 수, Recall = TP / 전체 정답 수
   - 예측이 없으면 Precision을 0으로 처리하며 정답은 존재한다고 가정
   - 단일 매칭 기준에서의 지표이며 AP나 mAP 계산은 아님

7. 실험 비교
   - 같은 예측·정답을 고정하고 confidence와 NMS IoU 기준만 변경
   - 첫째·둘째 조건은 confidence 효과, 첫째·셋째 조건은 NMS 임계값 효과 비교
   - 학습을 수행하지 않으며 이번 예시 결과로 다른 데이터의 성능을 단정할 수 없음

8. 파일과 결과 표
   - Path.resolve().parent: 현재 소스 폴더, mkdir(exist_ok=True): 출력 폴더 생성
   - read_csv(): 표 읽기, to_numpy(): 배열 변환, tolist(): Python 리스트 변환
   - DataFrame(): 결과 표 생성, to_csv(index=False): 행 번호 없이 CSV 저장
   - round(4): 반올림, to_string(index=False): 행 번호 없이 표 출력
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# Tensor 연산과 조건에 맞는 인덱스 선택
import torch
# CSV 읽기와 결과 표 저장
import pandas as pd
# 박스 간 IoU 계산과 클래스별 NMS
from torchvision.ops import box_iou, batched_nms
# 터미널 위치와 관계없이 현재 소스 폴더를 기준으로 사용
BASE=Path(__file__).resolve().parent
# 결과를 저장할 outputs 폴더 생성
out=BASE/"outputs";out.mkdir(exist_ok=True)
p=pd.read_csv(BASE/"data/boxes_fixture.csv");g=pd.read_csv(BASE/"data/boxes_gt.csv")
# 예측 좌표를 (N,4) 실수 Tensor로 변환. p는 예측 표, g는 정답 표
boxes=torch.tensor(p[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
# 예측별 confidence를 실수 Tensor로 변환
scores=torch.tensor(p.confidence.to_numpy(),dtype=torch.float32)
# 예측별 클래스 번호를 정수 Tensor로 변환
classes=torch.tensor(p.class_id.to_numpy(),dtype=torch.int64)
# 모든 정답 박스의 xyxy 좌표 준비
gb=torch.tensor(g[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
# 설정별 결과를 모을 리스트
rows=[]
# confidence와 NMS 임계값의 세 조합을 비교
for conf,nms_iou in [(0.25,0.5),(0.5,0.5),(0.25,0.95)]:
    # confidence를 통과한 원본 인덱스 추출
    ids=torch.where(scores>=conf)[0]
    # >>> A02
    # 요구: 같은 클래스끼리만 중복을 제거하는 batched_nms를 적용하세요.
    # 반환 index는 ids로 줄인 배열 기준이므로 다시 원본 index로 바꿉니다.
    # TODO A02: 위 요구사항의 핵심 코드를 작성하세요.
    # 좌표·점수·클래스를 같은 ids로 추려 순서를 일치시킨 뒤 클래스별 NMS 수행
    selected = batched_nms(
        boxes[ids],
        scores[ids],
        classes[ids],
        iou_threshold=nms_iou
    )
    # 축소 배열 기준 위치를 원본 위치로 변환: ids=[0,2,3], selected=[0,2] → [0,3]
    keep = ids[selected]
    # <<< A02
    # 실험 조건마다 사용한 정답과 TP 개수 초기화
    used=set();tp=0
    # NMS가 남긴 예측을 점수 내림차순으로 처리
    for i in keep.tolist():
        # boxes[[i]]는 (1,4)를 유지하며 반환 IoU 행렬의 첫 행 선택
        ovs=box_iou(boxes[[i]],gb)[0]
        # 같은 클래스이고 아직 사용하지 않은 정답만 후보로 선택
        eligible=[j for j in range(len(g)) if int(classes[i])==int(g.iloc[j].class_id) and j not in used]
        # IoU가 높은 후보부터 검사. 음수 정렬 키를 사용하여 내림차순 구현
        for j in sorted(eligible,key=lambda j:-float(ovs[j])):
            # 매칭 기준을 만족하면 정답 사용 기록, TP 증가, 현재 예측의 탐색 종료
            if ovs[j]>=0.5:used.add(j);tp+=1;break
    # 남은 예측 중 연결 실패는 FP, 정답 중 연결되지 않은 것은 FN
    n=len(keep);fp=n-tp;fn=len(g)-tp
    # 이번 설정과 평가 결과 기록. 예측이 0개일 때 Precision의 0 나눗셈 방지
    rows.append(dict(conf=conf,nms_iou=nms_iou,kept=n,tp=tp,fp=fp,fn=fn,precision=tp/n if n else 0,recall=tp/len(g)))
# 결과를 outputs/nms_comparison.csv로 저장하고 반올림한 표를 화면에 출력
r=pd.DataFrame(rows);r.to_csv(out/"nms_comparison.csv",index=False);print(r.round(4).to_string(index=False))
