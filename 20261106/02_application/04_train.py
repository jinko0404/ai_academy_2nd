"""실습 04. API 학습과 같은 설정의 CLI 명령을 기록합니다."""
"""
[실습 04 - A04 기본 개념: YOLO 학습 설정과 API·CLI 기록]

1. 객체 탐지와 YOLO
   - 객체 탐지는 물체의 클래스와 이미지 안 박스 위치를 함께 예측하는 작업
   - ultralytics.YOLO는 모델 생성과 학습·검증·추론을 제공하는 인터페이스
   - 이 파일은 2 epoch 학습으로 데이터와 학습 과정의 연결을 확인

2. 모델 구조와 가중치
   - yolo26n.yaml: 모델 구조 설정이며 학습된 가중치 파일이 아님
   - 이 코드는 구조에서 모델을 생성하고 pretrained=False로 학습
   - .pt: 학습된 가중치 등 모델 상태를 보관하는 파일 형식
   - 짧은 무작위 초기화 학습 결과를 사전학습 모델 성능으로 해석하면 안 됨

3. 데이터셋 YAML
   - train: 학습 이미지 경로, val: 검증 이미지 경로
   - names: 클래스 번호순 이름이며 square=0, circle=1, triangle=2
   - 이미지 폴더에 대응하는 labels 폴더에서 YOLO TXT 정답을 읽음
   - yaml.safe_dump(): 딕셔너리를 YAML 문자열로 변환
   - write_text(encoding="utf-8"): 설정 문자열을 파일로 저장
   - 데이터셋 YAML과 모델 구조 YAML은 역할이 서로 다름

4. API와 CLI
   - API: Python에서 model.train(...)으로 학습 실행
   - CLI: 터미널에서 yolo detect train ... 명령으로 학습 실행
   - 공정한 비교를 위해 데이터·모델·seed·학습 옵션을 동일하게 지정
   - 결과 폴더 이름만 smoke와 cli_smoke로 구분
   - 이 코드는 CLI 명령을 텍스트로 저장할 뿐 자동으로 실행하지 않음

5. 주요 학습 옵션
   - epochs=2: 학습 데이터를 두 번 사용
   - imgsz=160: 입력 크기 기준, batch=4: 배치 크기
   - workers=0: 별도 데이터 로딩 worker 프로세스를 사용하지 않음
   - device="cpu": CPU 사용, amp=False: 자동 혼합 정밀도 사용 안 함
   - optimizer="SGD": optimizer 종류 고정, lr0=0.01: 초기 학습률
   - seed=42: 난수 설정의 기준값으로 사용
   - 같은 seed라도 버전·장치·다른 설정이 다르면 완전히 같은 결과가 보장되지는 않음

6. 저장 옵션과 산출물
   - project와 name으로 출력 폴더를 지정하여 outputs/smoke에 저장
   - exist_ok=True: 기존 폴더 이름을 사용할 수 있게 하며 일부 파일이 덮어써질 수 있음
   - plots=False: 학습 관련 그래프 생성 비활성화
   - results.csv: epoch별 손실과 지표, args.yaml: 실행 설정 확인에 사용
   - weights/best.pt와 last.pt: 최적 상태와 마지막 학습 상태의 체크포인트
   - best 선택은 라이브러리의 검증 기준을 따르며 직접 설정한 valid_loss 비교가 아님

7. smoke 학습과 평가 범위
   - 데이터 읽기·순전파·손실·역전파·갱신의 연결이 동작하는지 점검하는 짧은 실행
   - 실행 성공만으로 충분한 학습이나 좋은 탐지 품질을 입증할 수 없음
   - 낮은 성능도 짧은 무작위 초기화 학습에서는 나올 수 있음
   - train은 학습, val은 검증에 사용하며 test 경로는 전달하지 않음

8. 실행 진입점과 CPU 설정
   - __name__ == "__main__": 직접 실행할 때만 학습 코드 수행
   - Windows worker나 다른 모듈의 import로 학습이 반복 실행되는 것을 방지
   - torch.set_num_threads(1): PyTorch CPU 연산 스레드 수 설정
   - 연산 스레드 수와 데이터 로더 workers는 서로 다른 설정

9. 경로 관리
   - Path.resolve().parent: 현재 소스 폴더를 기준 경로로 사용
   - mkdir(exist_ok=True): 폴더가 없으면 생성
   - str(path): Path 객체를 문자열로 변환하여 라이브러리에 전달
   - CLI 문자열의 경로를 큰따옴표로 감싸 공백이 포함된 경로를 하나의 인자로 표현
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# YAML 설정 문자열 생성과 PyTorch 실행 환경 설정
import yaml,torch

# YOLO 모델 생성과 학습 인터페이스
from ultralytics import YOLO

# 현재 소스 파일의 폴더를 기준 경로로 지정
BASE=Path(__file__).resolve().parent

# 직접 실행한 경우에만 아래 학습 과정 수행
if __name__=="__main__":
    # CPU 연산 스레드 수를 1개로 설정
    torch.set_num_threads(1)

    # 결과 저장 상위 폴더 생성
    out=BASE/"outputs";out.mkdir(exist_ok=True)

    # train·val 이미지 경로와 라벨 번호순 클래스 이름 정의
    config={"train":str(BASE/"data/images/train"),"val":str(BASE/"data/images/val"),"names":["square","circle","triangle"]}

    # 데이터셋 설정을 UTF-8 YAML로 저장하며 같은 파일이 있으면 덮어씀
    path=out/"dataset_active.yaml";path.write_text(yaml.safe_dump(config),encoding="utf-8")

    # 사전학습 .pt가 아닌 모델 구조 YAML에서 모델 생성
    model=YOLO("yolo26n.yaml")
    # >>> A04
    # 요구: 2 epoch smoke 학습을 실행하세요. seed·imgsz·batch·optimizer 조건을 고정합니다.
    model.train(
        # 위에서 생성한 데이터셋 YAML 경로
        data=str(path),
        # 전체 학습 데이터를 2회 반복
        epochs=2,
        # 학습 입력 크기 기준
        imgsz=160,
        # 배치 크기를 4로 고정
        batch=4,
        # 별도 데이터 읽기 worker 프로세스 비활성화
        workers=0,
        # CPU에서 학습
        device="cpu",
        # 자동 혼합 정밀도 비활성화
        amp=False,
        # 사전학습 가중치를 사용하지 않는 조건 명시
        pretrained=False,
        # optimizer 자동 선택 대신 SGD 사용
        optimizer="SGD",
        # 초기 학습률 지정
        lr0=0.01,
        # 라이브러리의 난수 시드 기준값 지정
        seed=42,
        # 그래프 등 학습 시각화 생성 비활성화
        plots=False,
        # 출력 상위 경로를 outputs로 지정
        project=str(out),
        # API 실행 결과를 outputs/smoke에 저장
        name="smoke",
        # 기존 smoke 폴더를 사용하도록 허용
        exist_ok=True,
    )
    # <<< A04
    # API와 동일한 명시 설정의 CLI 명령 생성. 결과 폴더만 cli_smoke로 분리
    # 문자열을 만드는 작업이므로 이 줄만으로 CLI 학습이 실행되지는 않음
    command=f'yolo detect train model=yolo26n.yaml data="{path}" epochs=2 imgsz=160 batch=4 workers=0 device=cpu amp=False pretrained=False optimizer=SGD lr0=0.01 seed=42 plots=False project="{out}" name=cli_smoke exist_ok=True'

    # 필요할 때 복사하여 실행할 CLI 명령을 UTF-8 텍스트 파일로 저장
    (out/"same_settings_cli.txt").write_text(command,encoding="utf-8")

    # API 체크포인트 폴더 안내. 실제 파일 생성 여부는 해당 폴더에서 확인
    print("best/last:",out/"smoke/weights", "CLI 비교는 선택 실행입니다.")
