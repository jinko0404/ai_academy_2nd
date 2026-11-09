"""실습 06: 배포 흐름의 설계 기록입니다. ONNX·TensorRT 엔진을 실행하지 않습니다."""
"""
[실습 06 - B07 기본 개념: 모델 배포 흐름과 검증 상태 기록]

1. 배포 계획
   - 모델을 실제 사용할 환경에서 실행하기 위해 준비·변환·검증하는 절차
   - 이 파일은 절차를 기록하는 실습이며 모델 변환이나 추론을 실행하지 않음
   - DESIGN_ONLY는 설계만 작성했다는 상태이며 배포 완료를 뜻하지 않음

2. 모델과 입력 계약
   - 배포 대상 모델의 구조·가중치·입력 크기·자료형·전처리 등을 먼저 정의
   - 모델이 지원하는 내보내기 방법과 연산 지원 여부를 확인해야 함
   - 이 계획은 Tesseract 실행 프로그램 자체를 ONNX로 변환하는 예제가 아님

3. ONNX 내보내기와 검증
   - ONNX는 모델의 연산 그래프와 가중치 등을 전달하는 형식
   - 내보내기 후 구조와 입출력 규격, 원래 모델과의 출력 차이 등을 확인
   - 파일 생성 성공만으로 변환 후 정확성이 검증되지는 않음

4. 대상 장치에서 엔진 build
   - 대상 환경에 맞춰 TensorRT 실행 엔진을 구성하는 단계
   - 장치·버전·지원 연산·정밀도 등의 조건을 기록하고 호환성을 확인
   - 엔진 생성·검증 전에는 실제 생성물의 해시를 기재할 수 없음

5. Runtime 추론과 검증
   - Runtime은 준비된 엔진을 실제 입력으로 실행하는 단계
   - 입력 전처리·출력 후처리·정확도·지연시간 등을 확인해야 함
   - 엔진 build 성공과 실제 업무 성능 검증 성공은 서로 다른 단계

6. None과 False의 의미
   - None은 값이 없거나 아직 확인하지 않았음을 표현하며 JSON에서는 null
   - latency_ms=None은 미측정이며 실행이 0ms 걸렸다는 뜻이 아님
   - checked_on_hardware=False는 실제 장치에서 확인하지 않았다는 표시
   - 장치명·TensorRT 버전·엔진 해시도 실제 확인한 뒤에만 채워야 함

7. 리스트와 JSON 저장
   - stages는 처리 순서대로 문자열을 담은 리스트
   - plan은 단계와 검증 상태를 함께 보관하는 딕셔너리
   - json.dumps(): Python 객체를 JSON 문자열로 변환
   - ensure_ascii=False: 한글 유지, indent=2: 들여쓰기 2칸
   - write_text(): UTF-8 파일로 저장하며 같은 이름의 기존 결과는 덮어씀

8. 경로와 화면 출력
   - Path.resolve().parent: 현재 소스 파일의 폴더를 기준 경로로 사용
   - ' -> '.join(stages): 단계 사이에 화살표를 넣어 표시용 문자열 생성
   - 순서를 출력하거나 JSON을 저장하는 것은 각 단계를 실제 실행하는 작업이 아님
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# 배포 계획을 JSON 문자열로 변환
import json

# 터미널 실행 위치와 관계없이 현재 소스 폴더를 기준으로 사용
base = Path(__file__).resolve().parent
# TODO B07
# 모델 준비부터 실제 실행 검증까지의 순서를 기록. 각 항목은 실행 코드가 아닌 설명 문자열
stages = [
    '모델',
    'ONNX 내보내기/검증',
    '대상 장치에서 엔진 build',
    'Runtime 추론/검증',
]
# TODO 끝 B07
# 현재는 설계만 작성했으므로 미확인 정보와 미측정 시간은 None, 장치 검증은 False로 유지
plan = {'stages': stages, 'status': 'DESIGN_ONLY', 'target_device': None, 'tensorrt_version': None, 'engine_sha256': None, 'latency_ms': None, 'checked_on_hardware': False}
# 순서와 검증 상태를 UTF-8 JSON으로 저장. Python None은 null로 변환
(base / 'p06_deployment_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
# 단계별 설명을 화살표로 연결해 터미널에 출력
print(' -> '.join(stages))
# 실제 변환·엔진 실행이 아니라는 실습 범위를 명시
print('DESIGN_ONLY: 오늘은 설계 기록. Tesseract 자체를 ONNX로 변환하는 예제가 아닙니다.')
