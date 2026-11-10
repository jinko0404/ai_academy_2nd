"""
[Day17 - A07 기본 개념: 명시적 데이터 초기화]

1. argparse
   - 명령행 인자를 읽으며 --confirm-reset이 있어야 초기화를 진행함
   - action="store_true"는 옵션이 있으면 True, 없으면 False를 만듦

2. seed와 작업 데이터
   - knowledge.seed.json은 초기 기준 데이터, knowledge.json은 현재 작업 상태임
   - 초기화는 기준 데이터로 돌아가는 작업이며 서버 재시작과 다름
   - 서버를 끈 뒤 필요한 경우에만 실행하며 재시작 보존 실습 중에는 사용하지 않음

3. 검증·백업·교체
   - 초기 데이터 형식을 검사한 뒤 기존 파일을 시간 이름으로 백업함
   - 임시 파일을 완성한 뒤 교체하고 finally에서 잔여 임시 파일을 정리함

4. __name__
   - 직접 실행할 때만 main()을 호출하며 import만 하면 초기화하지 않음
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import shutil


def main() -> None:
    # 명령행 옵션과 도움말 정의
    parser = argparse.ArgumentParser(description="현재 폴더의 학습 데이터 초기화")
    parser.add_argument("--confirm-reset", action="store_true", help="기존 작업 데이터 초기화를 명시적으로 승인")
    # 실제로 입력한 명령행 옵션을 읽음
    args = parser.parse_args()
    if not args.confirm_reset:
        parser.error("서버를 먼저 종료하세요. 초기화가 필요할 때만 --confirm-reset을 붙입니다.")
    # 실행한 터미널 위치가 아닌 이 파일의 폴더를 기준으로 경로 결정
    base = Path(__file__).resolve().parent
    data = base / "data" / "knowledge.json"
    seed = base / "data" / "knowledge.seed.json"
    # 기준 파일을 문자열로 읽고 Python 객체로 변환
    obj = json.loads(seed.read_text(encoding="utf-8"))
    # 전체 dict, entries 목록, 정수 next_id 구조를 확인
    if not isinstance(obj, dict) or not isinstance(obj.get("entries"), list) or not isinstance(obj.get("next_id"), int):
        raise ValueError("초기 데이터 형식이 올바르지 않습니다. 현재 파일은 유지합니다.")
    if data.exists():
        # 백업 파일 이름에 시각과 마이크로초를 포함
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = data.with_name(f"knowledge.backup_{stamp}.json")
        # 기존 파일과 메타데이터를 백업 경로로 복사
        shutil.copy2(data, backup)
        print("백업:", backup)
    # 초기화 데이터도 임시 파일을 거쳐 저장
    temp = data.with_suffix(".reset.tmp")
    try:
        temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(data)
    finally:
        temp.unlink(missing_ok=True)
    print("초기화 완료:", data)
    print("일일프로젝트의 재시작 확인 중에는 이 명령을 실행하지 않습니다.")


# python reset_data.py --confirm-reset로 직접 실행한 경우에만 호출
if __name__ == "__main__":
    main()
