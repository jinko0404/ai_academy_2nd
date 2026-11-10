"""
[01_basic / 02_read / storage.py 기본 개념]

1. JSON 영속화
   - read_data는 파일을 dict로 읽고 write_data는 변경된 전체 상태를 저장
   - 파일 없음·JSON 손상은 예외로 알리며 임의의 빈 상태로 덮어쓰지 않음
2. 잠금
   - threading.Lock은 같은 Python 프로세스 안의 스레드 간 접근을 보호
   - 호출자가 동일 LOCK으로 읽기·수정·저장을 함께 묶어야 함
3. 임시 파일과 교체
   - 원본을 먼저 비우지 않고 임시 파일을 완성한 뒤 Path.replace로 교체
   - finally로 남은 임시 파일을 정리하며 모든 장애에 대한 복구를 보장하는 DB는 아님
4. 경로와 인코딩
   - __file__ 기준으로 교시별 data/knowledge.json을 사용하며 다른 교시와 공유하지 않음
   - UTF-8과 ensure_ascii=False로 한글 저장, indent=2로 읽기 쉬운 들여쓰기
"""

"""제공 코드: 파일 접근 보조 함수. 호출자가 같은 LOCK으로 읽기~저장을 묶습니다."""
# JSON 문자열과 Python 객체의 변환 도구
import json
# 파일 경로를 운영체제에 맞게 다루는 도구
from pathlib import Path
# 단일 프로세스의 스레드 간 잠금 도구
from threading import Lock

# 터미널 위치 대신 현재 소스 파일이 있는 폴더를 경로 기준으로 사용
BASE = Path(__file__).resolve().parent
# 이 교시에서만 사용할 현재 작업 데이터 파일 경로
DATA = BASE / "data" / "knowledge.json"
# 모든 파일 접근 호출자가 공유할 잠금 객체 생성
LOCK = Lock()


# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def read_data() -> dict:
    # 파일이 없거나 JSON이 손상되면 예외를 알립니다. 빈 목록으로 덮어쓰지 않습니다.
    # UTF-8 파일을 열고 with 블록 종료 시 자동으로 닫음
    with DATA.open("r", encoding="utf-8") as f:
        # 열린 UTF-8 JSON 파일에서 전체 상태를 읽어 반환
        return json.load(f)


# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def write_data(data: dict) -> None:
    # 원본부터 비우지 않고 같은 폴더의 임시 파일을 완성한 다음 교체합니다.
    # 단일 프로세스의 같은 LOCK 아래에서만 호출합니다. DB 트랜잭션은 아닙니다.
    # 원본과 같은 폴더에 임시 저장 파일 경로 생성
    temp = DATA.with_suffix(".tmp")
    # 정상 처리·예외 여부와 관계없이 finally 정리가 수행되도록 구성
    try:
        # UTF-8 파일을 열고 with 블록 종료 시 자동으로 닫음
        with temp.open("w", encoding="utf-8") as f:
            # 전체 상태를 한글이 보존되는 JSON으로 임시 파일에 기록
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 완성한 임시 파일로 원본을 교체
        temp.replace(DATA)
    # 앞선 파일 작업에서 오류가 나도 자원·임시 파일 정리를 시도
    finally:
        # 남은 임시 파일 정리. 이미 없으면 오류 없이 진행
        temp.unlink(missing_ok=True)
