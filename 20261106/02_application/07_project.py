"""실습 07~08. Tracker 두 종류를 독립 인스턴스로 비교합니다."""
"""
[실습 07~08 - A07·A08 기본 개념: Tracker 비교와 결과 기록]

1. 탐지와 추적
   - 탐지는 프레임마다 물체의 클래스·박스·점수를 예측하는 작업
   - 추적은 프레임 사이의 탐지를 연결하여 같은 개체에 ID를 부여하는 작업
   - class_id는 종류, track_id는 영상 안에서 연결된 개체의 번호
   - ID는 실제 신원이나 영구 번호가 아니며 누락·가림 등으로 바뀔 수 있음

2. ByteTrack과 BoT-SORT 비교
   - bytetrack.yaml과 botsort.yaml로 서로 다른 추적 설정을 선택
   - 같은 demo.pt, 입력 영상, confidence, 입력 크기, CPU 조건을 유지
   - 각 조건에서 새 YOLO 인스턴스를 생성하여 이전 추적 상태와 분리
   - 추적 설정 파일 안의 세부 옵션도 결과에 영향을 주므로 설정 파일을 함께 기록
   - 특정 이름의 tracker가 항상 더 우수하다고 이 실험만으로 단정할 수 없음

3. A07: track()과 stream
   - source: 입력 영상, tracker: 현재 반복문의 추적 설정 파일
   - stream=True: 프레임별 Results를 순차 반환하는 제너레이터 방식
   - 제너레이터를 for문에서 소비하면서 추론·추적이 수행됨
   - persist=False는 새 파일 스트림의 시작이며 같은 영상 내부 프레임은 연속 처리
   - conf=0.25, imgsz=160, device="cpu" 조건으로 실행

4. 프레임과 추적 기록
   - enumerate(results): 0부터 처리 프레임 번호 부여
   - boxes.xyxy: 원본 픽셀 좌표, boxes.cls: 클래스, boxes.conf: 점수
   - boxes.id: 추적 ID이며 없으면 None일 수 있음
   - frame_counts는 0건 프레임도 포함하며 rows는 ID가 있는 박스마다 한 행
   - track_rows는 박스 기록 행 수이지 고유 ID 개수나 물체 수가 아님

5. A08: tracker별 CSV
   - frame, track_id, class_id, confidence, x1, y1, x2, y2 순서로 기록
   - Path(name).stem: bytetrack.yaml에서 bytetrack처럼 확장자 제외 이름 추출
   - tracker 이름별 파일로 저장하여 두 실험의 기록이 섞이지 않도록 구분
   - DataFrame의 columns로 빈 기록에도 열 이름 유지
   - index=False는 표의 행 번호를 별도 열로 저장하지 않는 옵션

6. 영상 출력
   - VideoWriter는 MJPG 코덱, 10 fps, (너비 160, 높이 128)로 고정
   - result.plot(): 박스·ID 등이 그려진 BGR 영상 배열 반환
   - 출력 크기와 실제 프레임 크기가 일치해야 하며 입력 영상도 직접 확인 필요
   - 입력 fps를 자동으로 읽는 코드가 아니므로 다르면 재생 속도가 달라질 수 있음
   - release(): 영상 기록을 마무리하고 자원 해제
   - 현재 코드는 writer의 열기·저장 성공 여부를 별도로 검사하지 않음

7. 시간 측정과 throughput
   - perf_counter(): 경과 시간 측정용 시계
   - 측정은 모델 로드 후 시작하며 writer 생성부터 영상 처리·해제까지 포함
   - decode, 추론, 추적, 시각화, encode와 구간 내 최초 실행 비용이 포함됨
   - 모델 파일 로드와 측정 뒤 CSV 저장 시간은 제외
   - throughput = 처리 프레임 수 / 경과 초이며 모델 추론만의 FPS가 아님
   - 출력 영상의 10 fps는 재생 속도 설정으로 throughput과 다른 값
   - 실행 순서·초기화·시스템 상태 때문에 한 번의 시간만으로 속도 우열을 단정할 수 없음

8. ID switch와 평가 한계
   - 고유 ID 수는 부여된 번호 종류의 수이며 ID switch 횟수와 다름
   - ID switch는 같은 실제 물체에 연결된 ID가 바뀌는 사건
   - 정확한 측정에는 프레임별 정답 개체와 예측의 대응을 추적해야 함
   - id_switch 열의 문구는 측정 결과가 아니라 추가 평가 필요성을 안내
   - 탐지·추적 정답과 비교하지 않으므로 이 코드에서 품질 지표를 산출하지 않음

9. 재현성과 결과 관리
   - 모델·영상·추적 설정·conf·입력 크기·장치·라이브러리 버전을 함께 기록
   - 새 영상마다 상태를 초기화하고 같은 조건으로 재실행하여 결과 비교
   - stream 방식이어도 CSV용 리스트는 메모리에 누적됨
   - 결과 파일은 같은 이름으로 덮어쓸 수 있으므로 비교용 실행 결과는 별도 보관
   - test 데이터는 사용하지 않으며 data/demo.pt가 실제 존재해야 실행 가능
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# 영상 처리 구간의 경과 시간 측정
from time import perf_counter

# OpenCV 영상 저장과 PyTorch CPU 실행 설정
import cv2,torch

# 추적 결과와 비교 요약을 표로 저장
import pandas as pd

# 모델 로드 및 영상 추적 인터페이스
from ultralytics import YOLO

# 현재 소스 파일의 폴더를 기준 경로로 사용
BASE=Path(__file__).resolve().parent

# outputs 폴더 생성과 CPU 연산 스레드 수 1개 설정
out=BASE/"outputs";out.mkdir(exist_ok=True);torch.set_num_threads(1)

# tracker별 요약 행을 모을 리스트
summaries=[]

# 동일 영상에 두 추적 설정을 각각 독립 실행
for name in ["bytetrack.yaml","botsort.yaml"]:
    # 같은 가중치로 새 모델 객체를 생성하여 이전 tracker 상태를 이어받지 않음
    model=YOLO(str(BASE/"data/demo.pt"))

    # 조건별 상세 기록 초기화 후 시간 측정 시작. 위 모델 로드 시간은 제외
    rows=[];frame_counts=[];start=perf_counter()

    # tracker 이름별 영상 작성기 생성. 출력 형식은 10 fps, 너비 160·높이 128로 고정
    writer=cv2.VideoWriter(str(out/(Path(name).stem+".avi")),cv2.VideoWriter_fourcc(*"MJPG"),10,(160,128))
    # >>> A07
    # 요구: demo.avi를 현재 name의 tracker로 track 하세요. 반환은 stream generator입니다.
    # TODO A07: 위 요구사항의 핵심 코드를 작성하세요.
    results = model.track(
        # 두 실험에서 같은 영상 사용
        source=str(BASE / "data/demo.avi"),
        # 현재 반복문 값에 따라 ByteTrack 또는 BoT-SORT 설정 선택
        tracker=name,
        # 프레임별 결과를 순차 반환
        stream=True,
        # 새로운 파일 스트림으로 시작하며 내부 프레임은 연속 추적
        persist=False,
        # 두 실험의 예측 점수 기준 고정
        conf=0.25,
        # 입력 크기 기준 고정
        imgsz=160,
        # 실행 장치 고정
        device="cpu",
    )
    # <<< A07
    # 제너레이터를 소비하면서 0부터 프레임 번호 부여
    for frame,r in enumerate(results):
        # 시각화 프레임을 영상에 기록하고 0건을 포함한 프레임별 박스 수 저장
        writer.write(r.plot());frame_counts.append([frame,len(r.boxes)])

        # ID가 할당된 결과에 대해서만 상세 추적 기록 생성
        if r.boxes.id is not None:
            # 같은 박스의 좌표·클래스·점수·ID를 CPU 리스트로 변환하여 함께 순회
            for box,cls,score,tid in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.id.cpu().tolist()):
                # *box로 네 좌표를 펼쳐 frame·ID·class·score·xyxy 순서로 기록
                rows.append([frame,int(tid),int(cls),score,*box])

    # 결과 영상 기록 종료 후 전체 처리 구간 경과 시간 계산
    writer.release();seconds=perf_counter()-start
    # >>> A08
    # 요구: tracker별 CSV에 frame·track_id·class_id·confidence·xyxy를 저장하세요.
    # TODO A08: 위 요구사항의 핵심 코드를 작성하세요.
    # rows의 기록 순서에 맞는 열 이름 지정. 추적 행이 없어도 CSV 열 이름 유지
    pd.DataFrame(
        rows,
        columns=["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"],
    # bytetrack_tracks.csv 또는 botsort_tracks.csv로 행 번호 없이 저장
    ).to_csv(out / (Path(name).stem + "_tracks.csv"), index=False)
    # <<< A08
    # 모든 처리 프레임의 박스 수를 tracker별 별도 CSV로 저장
    pd.DataFrame(frame_counts,columns=["frame","detections"]).to_csv(out/(Path(name).stem+"_frames.csv"),index=False)

    # 프레임 수·추적 행 수·시간·전체 처리량과 측정 범위를 요약
    # id_switch는 숫자를 계산하지 않으며 별도 GT 매칭이 필요하다고 명시
    summaries.append(dict(tracker=name,frames=len(frame_counts),track_rows=len(rows),seconds=seconds,throughput=len(frame_counts)/seconds,scope="decode+infer+track+plot+encode; first-run cost included",id_switch="별도 GT 매칭 필요"))

# 두 tracker의 요약을 outputs/tracker_comparison.csv로 저장하고 터미널에 출력
r=pd.DataFrame(summaries);r.to_csv(out/"tracker_comparison.csv",index=False);print(r.to_string(index=False))
