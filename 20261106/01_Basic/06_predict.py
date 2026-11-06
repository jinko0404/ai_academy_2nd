"""실습 06. 강사가 준비한 실제 도형 YOLO weight로 valid 이미지만 추론합니다."""
"""
[실습 06 - B07 기본 개념: YOLO 추론과 예측 결과 기록]

1. 추론과 학습·검증
   - model.predict(): 준비된 가중치로 클래스·위치·점수를 예측하며 추가 학습하지 않음
   - model.val()처럼 정답과 비교하여 mAP를 계산하는 코드는 아님
   - 탐지 개수만으로 정답 여부나 탐지 품질을 판단할 수 없음

2. 현재 사용하는 모델
   - 현재 경로는 p04_runs/smoke/weights/best.pt로 앞선 1 epoch 학습 결과
   - 원래 요구한 강사 준비 data/demo.pt와 다르므로 결과 출처를 구분해야 함
   - 이 실행은 smoke 모델의 추론 점검이며 충분히 학습된 모델의 성능을 뜻하지 않음

3. model.predict() 인자
   - source: 입력 이미지 폴더, conf=0.25: 결과에 남길 예측 점수 기준
   - imgsz=160: 추론 입력 크기 기준, device="cpu": CPU에서 실행
   - 점수가 기준을 통과해도 올바른 탐지라는 보장은 없음
   - val로 개발 중 결과를 확인하고 test는 최종 평가를 위해 남겨 둠

4. Results와 Boxes
   - results는 이미지별 Results 객체를 담음
   - result.path: 입력 경로, result.boxes: 해당 이미지의 탐지 정보
   - boxes.xyxy: (탐지 수, 4) 픽셀 좌표 [x1, y1, x2, y2]
   - boxes.cls: 클래스 번호, boxes.conf: 예측 점수
   - 클래스 번호의 의미는 model.names 매핑으로 확인
   - confidence를 잘 보정된 정답 확률로 단정하면 안 됨

5. 탐지 0건과 두 결과 표
   - counts: 이미지마다 한 행이며 탐지가 없어도 0으로 기록
   - rows: 탐지 박스마다 한 행이므로 탐지 0건이면 행이 추가되지 않음
   - 물체가 없거나, 놓쳤거나, 기준 미달일 수 있으므로 0건만으로 원인 확정 불가
   - 0건 이미지를 제외하면 처리 수와 결과 해석이 왜곡될 수 있음

6. Tensor 변환과 반복
   - cpu(): Tensor를 CPU로 이동, tolist(): Python 리스트로 변환
   - zip(): 좌표·클래스·점수의 같은 위치 항목을 묶어 순회
   - int(cls): 클래스 번호를 정수로 변환
   - *box: 네 좌표를 개별 리스트 원소로 펼침

7. 시각화와 저장
   - result.plot(): 탐지 박스 등이 그려진 BGR 이미지 배열 반환
   - cv2.imwrite(): 이미지 저장 후 성공 여부 반환. 현재 코드는 반환값을 검사하지 않음
   - DataFrame과 to_csv(index=False): 행 번호 없이 결과 CSV 저장
   - predictions는 박스별, counts는 이미지별 결과이므로 행 수가 다를 수 있음
   - 탐지가 없어도 predictions CSV는 지정한 열 이름을 유지

8. 경로와 실행 설정
   - Path.resolve().parent: 현재 소스 폴더, stem: 확장자를 제외한 파일 이름
   - set_num_threads(1): PyTorch CPU 연산 스레드 수 설정
   - 결과는 현재 소스 폴더에 저장하며 실제 이미지 저장 여부도 확인해야 함
"""

# 파일·폴더 경로 관리
from pathlib import Path
# 시각화 이미지 저장
import cv2
# CPU 연산 스레드 설정
import torch
# 예측 결과와 이미지별 개수를 표로 저장
import pandas as pd
# 학습된 YOLO 모델 로드와 추론
from ultralytics import YOLO
# 현재 소스 폴더를 기준 경로로 지정
BASE = Path(__file__).resolve().parent
# PyTorch CPU 연산 스레드를 1개로 설정
torch.set_num_threads(1)
# 현재는 강사 준비 demo.pt 대신 사용자가 지정한 1 epoch smoke 체크포인트를 사용
# 결과를 보고할 때 실제 사용한 모델 경로와 짧은 학습 이력을 명시해야 함
model = YOLO(str(BASE / "p04_runs/smoke/weights/best.pt"))
# rows는 박스별 예측, counts는 탐지 0건을 포함한 이미지별 개수를 저장
rows = []; counts = []
# >>> B07
# 요구: data/images/val을 conf=0.25, imgsz=160, CPU로 predict 하세요.
# source는 입력 경로, conf/imgsz/device는 선택 인자입니다.
# TODO B07: 위 요구사항의 핵심 코드를 작성하세요.
# 검증 이미지 폴더에 대해 예측을 수행하고 이미지별 결과를 반환받음
results = model.predict(
    # test가 아닌 val 이미지를 입력으로 사용
    source=str(BASE / "data/images/val"),
    # 결과에 남길 예측 점수 기준
    conf=0.25,
    # 입력 크기 기준
    imgsz=160,
    # GPU 자동 선택 없이 CPU 사용
    device="cpu"
)
# <<< B07
# 이미지별 예측 결과를 하나씩 처리
for result in results:
    # 원본 경로에서 확장자를 제외한 파일 이름 추출
    name = Path(result.path).stem
    # 탐지가 없어도 detections=0인 행을 기록
    counts.append({"image": name, "detections": len(result.boxes)})
    # 시각화 배열을 p06_원본이름.png로 저장. 실제 저장 성공 여부는 별도 확인 필요
    cv2.imwrite(str(BASE / ("p06_" + name + ".png")), result.plot())
    # 같은 박스의 좌표·클래스·점수를 묶음. 탐지가 0건이면 반복하지 않음
    for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        # *box로 네 좌표를 펼쳐 각 열에 대응시킴
        rows.append([name, int(cls), score, *box])
# 박스별 예측을 CSV 저장하며 탐지가 없어도 열 이름 유지
pd.DataFrame(rows, columns=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(BASE/"p06_predictions.csv", index=False)
# 탐지 0건 이미지를 포함한 이미지별 개수 CSV 저장
pd.DataFrame(counts).to_csv(BASE/"p06_counts.csv", index=False)
# 처리 이미지 수와 전체 탐지 박스 수를 구분해 출력
print("처리 이미지:", len(counts), "탐지 수:", len(rows))
