"""03 · @tool·Pydantic·JSON Schema · 기본필수 실행 파일.

기본 실행은 입력 ID 검사, 도구 등록, JSON Schema 출력, 실제 DB 조회 순으로 진행합니다.
--check로 DB 접속 없이 실습 함수 목록을 확인할 수 있습니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 같은 폴더의 tasks.py에 입력 검사와 도구 등록 함수가 있습니다.
import tasks


def main():
    # argparse로 터미널에서 전달된 실행 옵션을 읽습니다.
    ap=argparse.ArgumentParser(description='Day29 · 기존 DB 재사용 / CPU fallback 없음')
    ap.add_argument('--check',action='store_true',help='실제 DB/API 없이 순수 함수 TODO 자가 점검 (실제 실습 아님)')
    ap.add_argument('--serve',action='store_true',help='실습 08 FastAPI 실행(실제 DB·GPU·LLM)')
    args=ap.parse_args()
    if args.check:
        # tasks에서 이름이 '_'로 시작하지 않고 호출 가능한 항목을 출력하고 종료합니다.
        print('TODO 함수:',[name for name in dir(tasks) if not name.startswith('_') and callable(getattr(tasks,name))])
        print('실제 연동은 --check 없이 실행해야 합니다.')
        return
    # --serve는 이 파일에서 옵션 값만 읽어 둡니다.
    serve=args.serve

    # 예제 ID 63이 유효한 형식인지 확인하고 상태 조회 도구를 만듭니다.
    print('[입력 검사]',tasks.validate_id(63))
    registered=tasks.register_status_tool()
    # Pydantic 스키마를 JSON Schema 딕셔너리로 바꿔 필요한 입력의 이름과 자료형을 확인합니다.
    print('[Tool]',registered.name,registered.args_schema.model_json_schema())
    # invoke()에 인자 딕셔너리를 넘기면 실제 도구가 실행되어 63번 보고서의 상태를 조회합니다.
    print('[실제 조회]',registered.invoke({'report_id':63}))

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
