"""실습 04. 무작위 초기화 YOLO26의 1 epoch 학습 연결 점검. 성능 실험이 아닙니다."""
"""
[실습 04 - B05 기본 개념: YOLO 학습 연결 점검]

1. 객체 탐지와 YOLO
   - 객체 탐지는 이미지 안 물체의 클래스와 박스 위치를 함께 예측하는 작업
   - ultralytics.YOLO는 모델 생성과 학습·검증·추론 기능을 제공하는 인터페이스
   - 이 파일은 제공된 구조 설정으로 모델을 만들고 학습 실행 여부를 점검

2. 모델 구조 YAML과 사전학습 가중치
   - yolo26n.yaml은 모델 구조를 정의하는 설정 파일
   - 이 코드에서는 구조를 기반으로 무작위 초기화한 모델을 생성
   - 학습된 파라미터를 담은 사전학습 .pt 파일을 불러오는 것과 다름
   - 짧은 학습 결과로 사전학습 모델의 성능이나 충분히 학습된 성능을 판단하면 안 됨

3. 데이터셋 YAML
   - train: 학습 이미지 경로, val: 검증 이미지 경로
   - names: 클래스 번호 순서의 클래스 이름 목록
   - square=0, circle=1, triangle=2로 TXT 라벨의 번호와 일치해야 함
   - 이 폴더 구조에서는 images에 대응하는 labels 경로에서 YOLO TXT 라벨을 읽음
   - test 경로는 설정에 넣지 않으며 이번 학습 호출에 test를 전달하지 않음

4. yaml.safe_dump()와 write_text()
   - safe_dump(): Python 딕셔너리를 YAML 형식 문자열로 변환
   - write_text(): 문자열을 지정한 파일에 저장하며 기존 파일은 덮어씀
   - encoding="utf-8": 저장할 텍스트의 인코딩 지정
   - 데이터셋 YAML과 모델 구조 YAML은 서로 다른 용도의 파일

5. model.train()
   - Ultralytics 모델의 실제 학습 과정을 실행하는 메서드
   - PyTorch nn.Module의 train()처럼 모드만 전환하는 호출과 구분해야 함
   - data: 데이터셋 YAML 경로, epochs: 전체 학습 데이터 반복 횟수
   - imgsz=160: 학습 입력 크기 기준. 실제 전처리 모양은 라이브러리 설정에 따라 결정
   - batch=4: 배치 크기, device="cpu": CPU에서 실행
   - 생략한 옵션은 설치된 라이브러리의 기본 설정을 사용

6. 1 epoch 연결 점검(Smoke Test)
   - 데이터 읽기부터 모델 순전파·손실·역전파·갱신까지 실행되는지 확인하는 짧은 점검
   - epochs=1은 전체 학습 데이터를 한 번 사용하는 설정
   - 정상 실행은 학습 연결이 동작한다는 뜻이며 좋은 탐지 성능의 증명은 아님
   - train으로 파라미터를 학습하고 val은 검증용으로 사용

7. 실행 진입점과 CPU 설정
   - if __name__ == "__main__": 파일을 직접 실행할 때만 내부 코드 실행
   - 다른 모듈에서 import하거나 Windows worker가 파일을 불러올 때 학습 재실행 방지
   - torch.set_num_threads(1): PyTorch CPU 연산 스레드 수 설정
   - 스레드 수 설정은 데이터 로더 worker 수를 지정하는 것과 다름

8. 출력 정보와 저장 경로
   - model.task: 작업 종류, model.names: 클래스 매핑, model.device: 실행 장치
   - 마지막 print는 경로 문자열을 표시할 뿐 학습 결과 저장 위치를 설정하지 않음
   - project는 결과 상위 폴더, name은 그 아래 실행 폴더 이름을 지정
   - 현재 설정은 p04_runs/smoke를 사용하며 기존 폴더가 있으면 이름이 증가할 수 있음
   - 실제 저장 위치는 학습 로그에서 확인해야 함

9. Path와 경로 문자열
   - Path.resolve().parent: 현재 소스 파일의 폴더
   - / 연산자: 하위 파일·폴더 경로 연결
   - str(config_path): Path 객체를 라이브러리에 전달할 문자열 경로로 변환
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# 딕셔너리를 데이터셋 설정 YAML 문자열로 변환
import yaml

# CPU 연산 환경 설정에 사용하는 PyTorch
import torch

# YOLO 모델 생성과 학습 실행 인터페이스
from ultralytics import YOLO

# 터미널 위치와 관계없이 현재 소스 폴더를 기준으로 경로 생성
BASE = Path(__file__).resolve().parent
# Windows에서도 읽기 worker가 스크립트를 재실행하지 않도록 진입점을 구분합니다.
if __name__ == "__main__":
    # PyTorch CPU 연산에 사용하는 스레드를 1개로 설정
    torch.set_num_threads(1)

    # 학습·검증 이미지의 경로와 라벨 번호에 대응하는 이름 목록 정의
    # 모델 구조 설정이 아니라 어떤 데이터를 읽을지 지정하는 데이터셋 설정임
    config = {"train": str(BASE/"data/images/train"), "val": str(BASE/"data/images/val"),
              "names": ["square", "circle", "triangle"]}

    # 학습 함수에 전달할 데이터셋 YAML의 저장 경로
    config_path = BASE / "p04_dataset.yaml"

    # 설정 딕셔너리를 YAML 문자열로 바꿔 UTF-8 파일로 저장
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    # 모델 구조 YAML에서 모델을 생성. COCO 사전학습 .pt를 불러오는 것은 아님
    model = YOLO("yolo26n.yaml")
    # >>> B05
    # 요구: config_path로 1 epoch, imgsz=160, batch=4, CPU 학습을 호출하세요.
    # 나머지 옵션은 제공한 그대로 유지하세요. test 경로는 전달하지 않습니다.
    # TODO B05: 위 요구사항의 핵심 코드를 작성하세요.
    # data: 위에서 저장한 YAML 경로, epochs=1: 전체 학습 데이터를 1회 사용
    # imgsz=160: 입력 크기 기준, batch=4: 배치 크기, device="cpu": CPU 실행
    # 실제 학습을 호출하며 나머지 생략 옵션은 라이브러리 기본값 사용
    model.train(
        data=str(config_path),
        epochs=1,
        imgsz=160,
        batch=4,
        device="cpu",
        # 학습 결과의 상위 폴더를 이 소스 폴더 아래 p04_runs로 지정
        project=str(BASE / "p04_runs"),
        # 상위 폴더 아래 실행 이름을 지정하여 기본 저장 위치를 p04_runs/smoke로 설정
        # weights, results.csv, args.yaml 등의 학습 산출물이 이 실행 폴더에 저장됨
        # exist_ok를 지정하지 않았으므로 기존 smoke가 있으면 smoke2 등으로 증가할 수 있음
        name="smoke"
    )
    # <<< B05
    # 학습 후 모델 작업 종류·클래스 이름 매핑·실행 장치 확인
    print("task:", model.task, "names:", model.names, "device:", model.device)

    # 안내 경로 출력. 이 print가 실제 저장 경로를 지정하는 것은 아님
    # 기본 지정 경로를 안내하며 이름이 증가한 경우 실제 위치는 학습 로그에서 확인
    print("학습 결과:", BASE/"p04_runs/smoke")
