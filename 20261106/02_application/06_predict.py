"""실습 06. 동일 모델·입력에서 conf만 바꾸고 고정 IoU=.5 매칭을 비교합니다."""
"""
[실습 06 - A06 기본 개념: confidence별 추론과 일대일 매칭]

1. 비교 실험
   - 모델 가중치, val 이미지, 입력 크기, 실행 장치와 정답 매칭 규칙을 고정
   - confidence 기준만 0.15, 0.25, 0.50으로 변경
   - 현재 모델은 2 epoch smoke 체크포인트이며 강사 준비 demo.pt와 다름
   - 추가 학습하지 않으며 test는 읽지 않음

2. confidence와 IoU
   - confidence는 예측 점수이며 정답 확률로 잘 보정되었다고 보장할 수 없음
   - conf는 결과에 남길 점수 기준
   - IoU는 두 박스의 교집합 면적 / 합집합 면적
   - 정답 매칭 IoU는 모든 조건에서 0.5로 고정
   - 높은 점수가 정확한 박스 위치나 올바른 클래스를 보장하지 않음

3. predict()와 stream=True
   - source: 입력 경로, imgsz: 입력 크기 기준, device: 실행 장치
   - conf=conf는 인자에 현재 반복문 변수의 값을 전달
   - stream=True는 이미지별 Results를 순차 반환하는 제너레이터 방식
   - 동영상만을 위한 옵션이 아니며 결과를 순회하면서 처리

4. 예측과 정답 좌표
   - orig_shape는 원본 (높이, 너비), boxes.xyxy는 원본 기준 픽셀 좌표
   - boxes.cls는 예측 클래스 번호, boxes.conf는 예측 점수
   - 라벨은 class_id cx cy bw bh의 정규화 중심·크기 형식
   - x 좌표는 이미지 너비, y 좌표는 이미지 높이를 곱해 픽셀로 복원
   - gt는 정답 좌표, classes는 같은 순서의 정답 클래스 목록

5. 일대일 매칭
   - 예측을 점수 내림차순으로 처리하며 stable=True로 동점의 원래 순서 유지
   - 같은 클래스의 미사용 정답만 후보로 선택
   - IoU가 높은 후보부터 검사하여 0.5 이상이면 하나에 연결
   - used는 정답의 중복 사용을 막고 break는 한 예측의 다중 연결을 막음

6. TP, FP, FN
   - TP는 연결된 예측, FP는 남은 예측, FN은 남은 정답
   - 정답 없는 이미지의 예측은 모두 FP이며 예측 없는 이미지의 정답은 모두 FN
   - 둘 다 없으면 세 개수에 추가되는 값은 모두 0
   - Precision=TP/(TP+FP), Recall=TP/(TP+FN)
   - 이미지별 지표 평균이 아니라 전체 개수를 합산한 후 비율 계산
   - 분모가 0이면 이 코드는 0으로 보고하며 AP·mAP를 계산하지 않음

7. 결과 저장과 해석
   - outputs/confidence_comparison.csv에 조건별 결과 저장
   - 기준을 높여도 Precision이 반드시 좋아지는 것은 아님
   - 모든 조건에서 탐지가 0이면 이번 범위에서 기준 변경 효과를 확인하기 어려움
   - 짧은 smoke 모델의 결과를 충분히 학습된 모델이나 새 데이터의 성능으로 일반화하지 않음

8. 경로와 자료 변환
   - Path.resolve().parent: 현재 소스 폴더, stem: 확장자를 제외한 파일 이름
   - read_text().splitlines(): 라벨을 줄별 문자열로 읽음
   - map(float, ...): 숫자 변환, cpu(): CPU 이동, tolist(): Python 리스트 변환
   - DataFrame과 to_csv(index=False): 결과 표를 행 번호 없이 저장
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# Tensor 연산 사용. cv2는 현재 직접 사용하지 않음
import torch,cv2
# 비교 결과 표 생성과 CSV 저장
import pandas as pd
# 예측·정답 박스 간 IoU 계산
from torchvision.ops import box_iou
# 모델 로드와 추론 실행
from ultralytics import YOLO
# 현재 소스 폴더를 기준 경로로 사용
BASE=Path(__file__).resolve().parent
# 결과 폴더가 없으면 생성
out=BASE/"outputs";out.mkdir(exist_ok=True)
torch.set_num_threads(1);model=YOLO(str(BASE/"outputs/smoke/weights/best.pt"));records=[]
# 위에서 같은 smoke 모델을 로드한 뒤 confidence 기준만 바꾸어 비교
for conf in [.15,.25,.50]:
    # 현재 조건의 전체 이미지 누적 TP·FP·FN 초기화
    tp=fp=fn=0
    # >>> A06
    # 요구: 같은 val 이미지 폴더를 현재 conf 값으로 추론하세요. stream=True를 사용합니다.
    # TODO A06: 위 요구사항의 핵심 코드를 작성하세요.
    # 같은 검증 폴더에 대해 현재 confidence로 추론
    results = model.predict(
        # test가 아닌 val 이미지 사용
        source=str(BASE / "data/images/val"),
        # 오른쪽 conf는 반복문에서 선택한 현재 값
        conf=conf,
        # 입력 크기 기준 고정
        imgsz=160,
        # CPU 실행으로 고정
        device="cpu",
        # 결과를 이미지별로 순차 반환
        stream=True
    )
    # <<< A06
    # 결과 제너레이터를 소비하며 이미지별 처리
    for result in results:
        # 원본 크기 조회, 이미지별 정답 좌표와 클래스 목록 초기화
        h,w=result.orig_shape;gt=[];classes=[]
        # 이미지와 같은 기본 이름의 정답 TXT를 줄별로 읽음
        for line in (BASE/"data/labels/val"/(Path(result.path).stem+".txt")).read_text().splitlines():
            # 정규화 중심·크기를 픽셀 xyxy로 복원하고 클래스 번호를 같은 순서로 기록
            cls,cx,cy,bw,bh=map(float,line.split());gt.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]);classes.append(int(cls))
        # 사용한 정답과 매칭 수 초기화. IoU 계산을 위해 예측은 CPU로 이동
        used=set();preds=result.boxes.cpu();matched=0
        # 점수가 높은 예측부터 처리하며 동점은 원래 순서 유지
        for i in torch.argsort(preds.conf,descending=True,stable=True).tolist():
            # 정답이 없으면 매칭 시도만 생략. 아래 FP 계산에는 예측이 그대로 포함
            if not gt:continue
            # (1,4) 예측 하나와 모든 정답의 IoU를 계산한 뒤 첫 행 추출
            ovs=box_iou(preds.xyxy[[i]],torch.tensor(gt,dtype=torch.float32))[0]
            # 같은 클래스·미사용 정답을 IoU 내림차순으로 검사
            for j in sorted([j for j in range(len(gt)) if classes[j]==int(preds.cls[i]) and j not in used],key=lambda j:-float(ovs[j])):
                # 기준 충족 시 정답 사용 기록, 매칭 수 증가 후 현재 예측의 탐색 종료
                if ovs[j]>=.5:used.add(j);matched+=1;break
        # 이미지별 매칭 수로 TP·FP·FN을 계산하여 전체 합계에 추가
        tp+=matched;fp+=len(preds)-matched;fn+=len(gt)-matched
    # 전체 이미지 처리 후 조건별 지표 기록. 분모가 0이면 0 처리
    records.append(dict(conf=conf,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0))
# 결과 표를 CSV로 저장하고 행 번호 없이 터미널에도 출력
r=pd.DataFrame(records);r.to_csv(out/"confidence_comparison.csv",index=False);print(r.to_string(index=False))
