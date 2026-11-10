"""
[Day17 - A07 기본 개념: JSON 파일 저장]

1. Path와 JSON
   - Path는 경로를 관리하며 __file__ 기준 경로로 실행 위치에 따른 혼동을 줄임
   - json.load는 파일에서 dict를 읽고 json.dump는 전체 상태를 파일에 기록함
   - ensure_ascii=False는 한글 유지, indent=2는 들여쓰기 2칸임

2. 임시 파일과 교체
   - 원본부터 비우지 않고 임시 파일을 완성한 뒤 replace로 교체함
   - finally는 성공·실패와 관계없이 실행되어 남은 임시 파일을 정리함
   - write_data는 항목을 추가하는 함수가 아니라 전달받은 전체 상태를 저장함

3. Lock
   - 호출자가 같은 프로세스의 LOCK으로 읽기부터 저장까지 묶어야 함
   - 여러 프로세스를 보호하거나 DB 트랜잭션을 제공하지는 않음
"""
# Python 객체와 JSON 파일 사이의 변환
import json
from pathlib import Path
from threading import Lock

# 이 파일의 절대 경로에서 상위 폴더를 구함
BASE = Path(__file__).resolve().parent
# 이 실습만 사용하는 저장 파일
DATA = BASE / "data" / "knowledge.json"
# main.py가 가져다 공유할 잠금 객체
LOCK = Lock()


def read_data() -> dict:
    # 파일이 없거나 JSON이 손상되면 예외를 알립니다. 빈 목록으로 덮어쓰지 않습니다.
    with DATA.open("r", encoding="utf-8") as f:
        # entries와 next_id를 포함한 전체 dict를 반환
        return json.load(f)


def write_data(data: dict) -> None:
    # 원본부터 비우지 않고 같은 폴더의 임시 파일을 완성한 다음 교체합니다.
    # 단일 프로세스의 같은 LOCK 아래에서만 호출합니다. DB 트랜잭션은 아닙니다.
    # knowledge.json과 같은 폴더의 knowledge.tmp
    temp = DATA.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 완성한 임시 파일로 저장 파일을 교체
        temp.replace(DATA)
    finally:
        # 교체 후 파일이 없어도 오류를 내지 않음
        temp.unlink(missing_ok=True)
