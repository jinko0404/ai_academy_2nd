"""실습 05. 실제 준비 모델을 valid에 평가합니다. test는 전달하지 않습니다."""
"""
[실습 05 - A05 기본 개념: 준비 모델의 검증과 탐지 지표 저장]

1. 학습과 검증
   - 학습은 손실과 기울기로 모델 파라미터를 갱신하는 과정
   - 검증은 준비된 가중치로 예측을 수행하고 정답과 비교하는 과정
   - 이 파일은 model.val()로 평가하며 추가 학습을 수행하지 않음
   - 데이터셋 YAML에 train 경로가 있어도 split="val"로 평가할 분할을 선택
   - test는 전달하지 않으므로 결과는 test 성능이 아님

2. 평가할 모델과 가중치
   - YOLO(...best.pt): 앞서 저장한 학습 체크포인트를 불러옴
   - 현재 파일은 outputs/smoke/weights/best.pt를 사용
   - 이 가중치는 2 epoch 연결 점검용 학습의 결과이며 충분한 학습 성능을 보장하지 않음
   - 모델 구조 YAML에서 무작위 초기화 모델을 만드는 것과 구분해야 함

3. 데이터셋 설정과 클래스 매핑
   - train·val: 이미지 폴더 경로, names: 클래스 번호순 이름
   - square=0, circle=1, triangle=2이며 모델과 라벨의 클래스 의미가 일치해야 함
   - safe_dump(): 딕셔너리를 YAML 문자열로 변환
   - write_text(): 설정 문자열을 파일로 저장. 기존 파일은 덮어씀

4. model.val()과 실행 장치
   - data: 데이터셋 YAML 경로, split="val": 검증 분할 선택
   - device="cpu": GPU 자동 선택을 막고 CPU로 검증
   - 현재 환경에서 발생했던 CUDA용 torchvision NMS 연산 오류를 피하기 위한 설정
   - CPU 지정은 패키지의 GPU 연산 지원 문제 자체를 해결하는 것은 아님
   - workers=0: 별도 데이터 로딩 worker 프로세스를 사용하지 않음
   - 나머지 생략한 옵션은 모델에 저장된 설정이나 라이브러리 기본값의 영향을 받을 수 있음

5. IoU와 AP
   - IoU = 예측·정답 박스의 교집합 면적 / 합집합 면적
   - AP는 예측 점수 순서에 따른 Precision·Recall 관계를 요약하는 탐지 지표
   - 평가에서는 클래스, 위치 일치와 중복 예측 등을 고려
   - mAP는 AP를 클래스 등에 걸쳐 평균한 값이며 Accuracy와 다름

6. mAP50과 mAP50-95
   - metrics.box.map50: IoU 0.50 기준의 mAP
   - metrics.box.map: IoU 0.50~0.95를 0.05 간격으로 적용한 mAP
   - 엄격한 위치 기준을 포함하는 mAP50-95는 일반적으로 mAP50보다 낮음
   - 0.7이라는 mAP를 이미지 70%를 맞혔다는 뜻으로 해석하면 안 됨

7. 클래스별 지표
   - ap_class_index: 클래스별 AP 결과 항목에 대응하는 실제 클래스 번호 목록
   - enumerate(): 결과 위치 i와 실제 클래스 번호 cls를 함께 가져옴
   - class_result(i): 결과 배열의 i번째 클래스에 대한 Precision, Recall, AP50, AP50-95
   - i는 결과 위치이고 cls는 클래스 번호이므로 역할을 구분해야 함
   - Precision: 예측 중 올바른 탐지 비율, Recall: 정답 중 찾은 비율
   - 클래스별 출력 행이 모든 클래스 번호를 반드시 포함한다고 가정하지 않음

8. JSON과 CSV 산출물
   - summary: 전체 mAP 두 종류와 평가 분할을 담은 딕셔너리
   - json.dumps(): 딕셔너리를 JSON 문자열로 변환
   - float(), int(): 지표를 Python 숫자로 변환하여 저장과 출력에 사용
   - validation_metrics.json: 전체 지표, class_metrics.csv: 클래스별 지표
   - to_csv(index=False): DataFrame 행 번호는 저장하지 않음

9. 경로와 실행 진입점
   - Path.resolve().parent: 현재 소스 폴더, mkdir(exist_ok=True): 폴더 생성
   - __name__ == "__main__": 직접 실행할 때만 평가 수행
   - set_num_threads(1): PyTorch CPU 연산 스레드 수 지정
   - 연산 스레드 수와 데이터 로딩 workers는 서로 다른 설정

10. 결과 해석의 범위
    - 실제 평가가 완료된 경우 이 모델과 이 val 분할에 대한 결과로 기록
    - 05_metrics.py의 가상 기록과 달리 이 파일은 실제 모델을 평가
    - 짧은 학습 모델의 낮은 지표만으로 코드 연결이 실패했다고 단정하지 않음
    - 새로운 데이터의 성능이나 독립적인 최종 test 성능을 보장하지 않음
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# JSON 결과 저장, YAML 설정 생성, PyTorch 실행 환경 설정
import json,yaml,torch

# 클래스별 지표를 표로 만들어 CSV 저장
import pandas as pd

# 학습된 모델 로드와 검증 실행 인터페이스
from ultralytics import YOLO

# 현재 소스 파일의 폴더를 기준 경로로 지정
BASE=Path(__file__).resolve().parent

# 직접 실행할 때만 아래 평가 과정 수행
if __name__=="__main__":
    # PyTorch CPU 연산 스레드를 1개로 지정
    torch.set_num_threads(1)

    # 결과 저장 폴더를 만들고 이미 있으면 그대로 사용
    out=BASE/"outputs";out.mkdir(exist_ok=True)

    # 데이터셋 이미지 경로와 클래스 번호에 대응하는 이름 목록 정의
    config={"train":str(BASE/"data/images/train"),"val":str(BASE/"data/images/val"),"names":["square","circle","triangle"]}

    # 데이터셋 설정을 YAML 문자열로 바꿔 UTF-8 파일로 저장
    path=out/"dataset_active.yaml";path.write_text(yaml.safe_dump(config),encoding="utf-8")

    # 앞선 2 epoch smoke 학습에서 저장한 best.pt를 로드
    # 해당 파일이 실제로 존재해야 하며 이번 평가에서는 가중치를 갱신하지 않음
    model=YOLO(str(BASE/"outputs/smoke/weights/best.pt"))
    # >>> A05
    # 요구: model.val에 data=path, split='val'을 전달하세요.
    # TODO A05: 위 요구사항의 핵심 코드를 작성하세요.
    metrics = model.val(
        # 위에서 저장한 데이터셋 YAML 경로
        data=str(path),
        # test가 아닌 검증 이미지 폴더를 평가 대상으로 선택
        split="val",
        # CUDA NMS 지원 오류를 피하도록 CPU에서 검증
        device="cpu",
        # 데이터 읽기용 별도 worker 프로세스 비활성화
        workers=0)
    # <<< A05
    # 반환된 검증 결과에서 전체 mAP를 Python 실수로 추출하고 분할 이름 기록
    summary={"map50":float(metrics.box.map50),"map50_95":float(metrics.box.map),"split":"val"}

    # 전체 지표를 들여쓰기 2칸의 JSON 문자열로 변환하여 UTF-8 저장
    (out/"validation_metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    # 클래스별 결과 행을 모을 리스트
    rows=[]

    # i는 지표 배열 안의 위치, cls는 실제 클래스 번호
    for i,cls in enumerate(metrics.box.ap_class_index):
        # 해당 결과 위치의 Precision, Recall, AP50, AP50-95를 순서대로 추출
        pr,rc,ap50,ap=metrics.box.class_result(i)
        # 실제 클래스 번호로 이름을 조회하고 지표를 Python 숫자로 변환해 기록
        rows.append([int(cls),model.names[int(cls)],float(pr),float(rc),float(ap50),float(ap)])

    # 클래스별 결과를 이름 있는 표로 구성하여 행 번호 없이 CSV 저장
    pd.DataFrame(rows,columns=["class_id","name","precision","recall","ap50","ap50_95"]).to_csv(out/"class_metrics.csv",index=False)

    # 전체 mAP와 평가한 분할을 터미널에 출력
    print(summary)
