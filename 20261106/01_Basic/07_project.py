"""실습 07~08. 실제 모델의 이미지 탐지와 영상 추적을 연결합니다."""
"""
[실습 07~08 - B08·B09 기본 개념: 이미지 탐지와 영상 추적 통합]

1. 탐지와 추적
   - 탐지는 이미지에서 물체의 클래스·박스·점수를 예측하는 작업
   - 추적은 연속 프레임의 탐지를 연결하고 물체에 추적 ID를 부여하는 작업
   - 클래스 번호는 물체 종류, track_id는 현재 영상에서 연결된 개체를 구분
   - 추적 ID는 실제 신원이나 영구 번호가 아니며 가림·오탐 등으로 바뀔 수 있음

2. 모델과 입력
   - data/demo.pt는 실습용으로 준비한 학습 가중치이며 파일이 실제 존재해야 함
   - val 이미지로 탐지하고 demo.avi로 추적하며 test는 읽지 않음
   - 이 파일은 학습이나 정답 기반 mAP 평가를 수행하지 않음
   - conf=0.25, imgsz=160, device="cpu" 조건으로 실행

3. Results와 Tensor
   - Results 객체는 입력별 예측 정보를 보관
   - boxes.xyxy: 원본 픽셀 좌표, boxes.cls: 클래스 번호, boxes.conf: 예측 점수
   - boxes.id: 추적 ID이며 할당된 ID가 없으면 None일 수 있음
   - cpu().tolist(): Tensor를 CPU로 옮겨 Python 리스트로 변환
   - zip(): 같은 위치의 좌표·클래스·점수·ID를 묶음
   - *box: 네 좌표를 개별 열 값으로 펼침

4. ByteTrack과 track()
   - ByteTrack은 탐지 박스를 프레임 사이에서 연결하는 추적 방법
   - tracker="bytetrack.yaml"로 해당 추적 설정을 지정
   - 새 모델 인스턴스로 새 영상의 추적을 시작하여 이전 상태와 분리
   - persist=False는 이 파일 스트림을 새로 시작하는 조건이며 매 프레임 초기화를 뜻하지 않음
   - 하나의 파일 내부 연속 프레임에서는 추적 상태를 이어 사용

5. stream=True와 프레임 처리
   - 전체 결과를 한꺼번에 모으지 않고 순차 반환하는 제너레이터 방식
   - enumerate(results): 결과를 소비하며 0부터 프레임 번호 부여
   - 여기의 frame은 처리 순서 번호이며 초 단위 시간이 아님
   - stream 방식이어도 tracks와 frame_rows는 처리한 기록을 메모리에 누적

6. 영상 메타데이터와 저장
   - VideoCapture: 영상 열기, get(): fps와 가로·세로 크기 확인
   - fps: 초당 프레임 수, size: (너비, 높이)
   - VideoWriter: 지정한 코덱·fps·크기로 결과 영상 기록
   - MJPG: 사용할 영상 코덱, result.plot(): 박스·ID 등이 그려진 BGR 배열
   - release(): 입력 또는 출력 영상 자원 해제
   - 이 코드는 정상 영상을 전제로 하며 열기·저장 성공을 별도로 검사하지 않음

7. 세 종류의 CSV
   - p07_predictions.csv: val 이미지의 박스별 예측
   - p07_tracks.csv: 영상의 프레임·ID별 추적 기록
   - p07_frames.csv: 탐지가 0개인 프레임도 포함한 프레임별 박스 수
   - 한 프레임에 물체가 여러 개면 tracks에 여러 행이 생길 수 있음
   - ID가 None이면 해당 프레임의 tracks 행은 추가하지 않음
   - 따라서 tracks 행 수와 전체 프레임 수는 서로 다른 값

8. 재실행 확인과 해석
   - 같은 모델·입력·설정으로 재실행하여 프레임 수, CSV, 영상을 비교
   - 영상이 저장되었다고 탐지·추적 품질이 좋다는 뜻은 아님
   - ID 유지 여부를 영상으로 확인하고 전환·누락은 별도로 관찰해야 함
   - 정답 추적 정보와 비교하지 않으므로 이 코드만으로 추적 정확도를 산출하지 않음

9. 경로와 표 저장
   - Path.resolve().parent: 현재 소스 폴더, name: 확장자 포함 파일 이름
   - DataFrame(..., columns=...): 기록 순서에 대응하는 열 이름 지정
   - to_csv(index=False): 표의 행 번호를 별도 열로 저장하지 않음
   - 같은 이름의 결과 파일은 재실행 시 덮어쓸 수 있으므로 비교할 파일은 별도 보관
"""

# 파일·폴더 경로 관리
from pathlib import Path
# 영상 메타데이터 읽기와 결과 영상 저장
import cv2
# PyTorch CPU 실행 설정
import torch
# 탐지·추적 기록을 CSV로 저장
import pandas as pd
# YOLO 모델 로드, 이미지 탐지와 영상 추적 실행
from ultralytics import YOLO
# 현재 소스 폴더를 모든 파일 경로의 기준으로 사용
BASE = Path(__file__).resolve().parent
# CPU 연산 스레드 수를 1개로 설정
torch.set_num_threads(1)
# 준비된 실습 가중치 로드. 파일이 없으면 이 단계에서 실행이 중단됨
model = YOLO(str(BASE / "data/demo.pt"))
# val 이미지의 박스별 예측 정보를 모을 리스트
rows = []
# val 이미지를 CPU로 추론. verbose=False로 상세 추론 로그를 줄임
for r in model.predict(source=str(BASE/"data/images/val"), conf=0.25, imgsz=160, device="cpu", verbose=False):
    # 좌표·클래스·점수를 같은 박스 순서로 순회. 탐지 0건이면 반복하지 않음
    for box, cls, score in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist()):
        # 파일 이름, 정수 클래스 번호, 점수와 네 좌표 기록
        rows.append([Path(r.path).name, int(cls), score, *box])
# 열 이름을 지정하여 이미지 탐지 기록 저장. 탐지 없는 이미지는 별도 행이 없음
pd.DataFrame(rows, columns=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]).to_csv(BASE/"p07_predictions.csv", index=False)
# 새 영상은 새 모델 인스턴스로 시작하여 이전 추적 상태를 이어붙이지 않습니다.
tracker = YOLO(str(BASE / "data/demo.pt"))
# 출력 영상 형식을 맞추기 위해 입력 영상을 열어 메타데이터 확인
cap = cv2.VideoCapture(str(BASE / "data/demo.avi"))
# 입력 영상의 초당 프레임 수 조회
fps = cap.get(cv2.CAP_PROP_FPS)
# VideoWriter가 요구하는 (너비, 높이) 순서로 크기 구성
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 메타데이터 확인용 입력 자원 해제. 실제 프레임 읽기는 아래 track에서 수행
cap.release()
# MJPG 코덱과 입력 fps·크기를 사용하여 추적 결과 영상 작성기 생성
writer = cv2.VideoWriter(str(BASE / "p07_tracked.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
# tracks는 ID별 상세 기록, frame_rows는 모든 처리 프레임의 박스 수 기록
tracks = []; frame_rows = []
# >>> B08
# 요구: demo.avi를 ByteTrack으로 읽는 stream=True 결과 generator를 만드세요.
# persist=False는 새로운 파일 스트림의 시작입니다. 같은 파일 내부 프레임은 이어집니다.
# TODO B08: 위 요구사항의 핵심 코드를 작성하세요.
results = tracker.track(
    # 새로 추적할 영상 파일
    source=str(BASE / "data/demo.avi"),
    # ByteTrack 설정 선택
    tracker="bytetrack.yaml",
    # 프레임별 결과를 순차 반환
    stream=True,
    # 새 파일 스트림으로 시작하되 파일 내부 프레임은 연속 추적
    persist=False,
    # 예측 점수 기준
    conf=0.25,
    # 모델 입력 크기 기준
    imgsz=160,
    # CPU에서 실행
    device="cpu",
)
# <<< B08
# 제너레이터를 소비하면서 0부터 처리 프레임 번호 부여
for frame, result in enumerate(results):
    # 박스·클래스·ID 등이 그려진 프레임을 영상에 추가
    writer.write(result.plot())
    # 탐지 0건인 프레임도 개수 0으로 기록
    frame_rows.append([frame, len(result.boxes)])
    # 추적 ID가 있을 때만 ID별 상세 기록 수행
    if result.boxes.id is not None:
        # 같은 객체의 좌표·클래스·점수·추적 ID를 함께 조회
        for box, cls, score, tid in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.id.cpu().tolist()):
            # 프레임과 정수 ID·클래스, 점수, xyxy 좌표를 순서대로 기록
            tracks.append([frame, int(tid), int(cls), score, *box])
# 결과 영상 기록을 마무리하고 파일 자원 해제
writer.release()
# >>> B09
# 요구: tracks를 지정한 열 이름의 CSV로 저장하세요. index=False를 사용합니다.
# tracks의 값 순서와 동일한 열 이름을 지정. 기록이 없어도 열 이름은 유지
pd.DataFrame(
    tracks,
    columns=["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"],
# index=False로 DataFrame 행 번호는 저장하지 않음
).to_csv(BASE / "p07_tracks.csv", index=False)
# <<< B09
# 모든 프레임의 박스 수를 별도 CSV로 저장
pd.DataFrame(frame_rows, columns=["frame", "detections"]).to_csv(BASE/"p07_frames.csv", index=False)
# 입력 형식과 실제 처리 프레임 수 출력
print("입력 fps/size:", fps, size, "처리 Frame:", len(frame_rows))
# 동일 조건 재실행과 산출물 비교 안내
print("실습 08: 설정을 바꾸지 말고 다시 실행해 Frame 수·CSV·영상을 확인하세요.")
