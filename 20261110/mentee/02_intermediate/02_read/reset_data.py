"""
[02_intermediate / 02_read / reset_data.py 기본 개념]

1. 명시적 데이터 초기화
   - 서버를 종료하고 --confirm-reset을 지정했을 때만 seed를 현재 데이터로 복원
   - 서버 시작 시 자동 호출하지 않으며 저장 지속성 검사 중에는 실행하지 않음
2. 입력 검증과 백업
   - seed의 dict·entries 목록·next_id 정수 구조를 확인한 뒤 작업
   - 기존 파일은 시간 기반 이름으로 백업하며 백업은 새 학습 데이터와 별도 보관
3. 파일 교체
   - 임시 파일을 완성한 후 교체하고 finally에서 임시 파일을 정리
   - 현재 데이터 삭제가 목적이므로 필요할 때만 사람이 명시적으로 실행
4. 실행 진입점
   - __name__ == '__main__'으로 직접 실행할 때만 main을 호출
"""

"""서버를 끈 상태에서만 명시적으로 초기화합니다. 서버 시작 시 호출하지 마세요."""
import argparse
# JSON 문자열과 Python 객체의 변환 도구
import json
# 파일 경로를 운영체제에 맞게 다루는 도구
from pathlib import Path
from datetime import datetime
import shutil


# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def main() -> None:
    # 명령행 옵션을 해석할 도구 생성
    parser = argparse.ArgumentParser(description="현재 폴더의 학습 데이터 초기화")
    # 파괴적인 초기화가 실수로 실행되지 않도록 명시 옵션 요구
    parser.add_argument("--confirm-reset", action="store_true", help="기존 작업 데이터 초기화를 명시적으로 승인")
    # 터미널에서 전달한 옵션을 읽음
    args = parser.parse_args()
    # 명시적인 초기화 승인 옵션이 없으면 작업 중단
    if not args.confirm_reset:
        # 서버 종료와 초기화 옵션 확인을 안내하고 실행 중단
        parser.error("서버를 먼저 종료하세요. 초기화가 필요할 때만 --confirm-reset을 붙입니다.")
    # 터미널 위치 대신 현재 소스 파일이 있는 폴더를 경로 기준으로 사용
    base = Path(__file__).resolve().parent
    # 이 교시에서만 사용할 현재 작업 데이터 파일 경로
    data = base / "data" / "knowledge.json"
    # 복원할 초기 상태가 들어 있는 seed 파일 경로
    seed = base / "data" / "knowledge.seed.json"
    # 초기 상태 JSON을 메모리 객체로 읽음
    obj = json.loads(seed.read_text(encoding="utf-8"))
    # 초기 파일의 최상위 구조와 필수 필드 자료형 검사
    if not isinstance(obj, dict) or not isinstance(obj.get("entries"), list) or not isinstance(obj.get("next_id"), int):
        # 초기 데이터가 잘못되었으면 현재 파일을 보존하고 오류 알림
        raise ValueError("초기 데이터 형식이 올바르지 않습니다. 현재 파일은 유지합니다.")
    # 현재 데이터가 있을 때만 백업 수행
    if data.exists():
        # 백업 이름 충돌을 줄이기 위해 마이크로초까지 시각 문자열 생성
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 현재 파일과 같은 폴더에 백업 이름 구성
        backup = data.with_name(f"knowledge.backup_{stamp}.json")
        # 기존 데이터를 메타데이터와 함께 백업 파일로 복사
        shutil.copy2(data, backup)
        # 작업 결과 또는 사용 주의사항을 터미널에 표시
        print("백업:", backup)
    # 초기화용 임시 파일 경로 구성
    temp = data.with_suffix(".reset.tmp")
    # 정상 처리·예외 여부와 관계없이 finally 정리가 수행되도록 구성
    try:
        # 초기 상태를 먼저 임시 파일에 완성
        temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 완성된 초기 상태 파일로 현재 작업 데이터를 교체
        temp.replace(data)
    # 앞선 파일 작업에서 오류가 나도 자원·임시 파일 정리를 시도
    finally:
        # 남은 임시 파일 정리. 이미 없으면 오류 없이 진행
        temp.unlink(missing_ok=True)
    # 작업 결과 또는 사용 주의사항을 터미널에 표시
    print("초기화 완료:", data)
    # 작업 결과 또는 사용 주의사항을 터미널에 표시
    print("일일프로젝트의 재시작 확인 중에는 이 명령을 실행하지 않습니다.")


# 다른 파일에서 import할 때는 초기화가 실행되지 않도록 진입점 분리
if __name__ == "__main__":
    # 명시적으로 실행한 초기화 절차 수행
    main()
