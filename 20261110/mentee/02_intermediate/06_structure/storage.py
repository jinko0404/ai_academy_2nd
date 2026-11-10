"""
[02_intermediate 실습 06 / storage.py 기본 개념]

1. 파일 접근 계층
   - read_data는 전체 JSON을 dict로 읽어 반환하고 write_data는 전달된 상태를 저장
   - 목록 추가·수정·삭제의 업무 처리는 main.py에서 하고 저장 함수는 전체 상태 기록
   - write_data의 반환은 None이며 HTTP 응답을 생성하지 않음

2. 데이터 파일과 경로
   - data/knowledge.json은 실제 데이터이며 storage.py는 이를 다루는 코드
   - __file__ 기준 경로로 현재 교시 파일만 사용
   - UTF-8로 읽고 쓰며 ensure_ascii=False로 한글 유지

3. 임시 파일 교체
   - 원본부터 비우지 않고 같은 폴더의 임시 JSON을 완성한 뒤 교체
   - finally에서 남은 임시 파일을 정리하며 오류를 빈 목록으로 숨기지 않음
   - 모든 전원 장애 복구나 디스크 영속성을 보장하는 DB 트랜잭션은 아님

4. 잠금 책임
   - 공유 LOCK은 같은 프로세스 스레드의 접근을 보호
   - 호출자가 읽기·수정·저장을 하나의 잠금으로 묶어야 함
   - 여러 worker나 외부 편집까지 보호하는 프로세스 간 잠금은 아님
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
    # 파일이 없거나 JSON이 손상되면 예외를 알립니다. 빈 목록으로 덮어쓰지 않습니다.
    # UTF-8 파일을 열고 with 블록 종료 시 자동으로 닫음
    with DATA.open("r", encoding="utf-8") as f:
        # 열린 UTF-8 JSON 파일에서 전체 상태를 읽어 반환
        return json.load(f)


# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def write_data(data: dict) -> None:
    # 원본부터 비우지 않고 같은 폴더의 임시 파일을 완성한 다음 교체합니다.
    # 단일 프로세스의 같은 LOCK 아래에서만 호출합니다. DB 트랜잭션은 아닙니다.
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
