"""07 · Tool 실패·권한·반복 상한 · 기본필수 실행 파일.

기본 실행은 실제 DB의 정상 조회 결과를 먼저 보여주고, 이어서 두 예제 ID의 검증 결과를 출력합니다.
--check를 사용하면 DB 접속 없이 실습 함수 목록을 확인할 수 있습니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 같은 폴더의 tasks.py에 ID 검증과 오류 분류 함수가 있습니다.
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

    # 기존 DB에서 63번 보고서의 상태를 읽기 전용으로 조회합니다. 접속 실패 시 여기서 예외가 발생합니다.
    from coursekit.db import fetch_report_status
    print('[실제 DB]',fetch_report_status(63))

    # 각 예제는 ID 변환만 실행합니다. -1의 ValueError는 오류 코드로 출력하고 다음 예제를 계속 검사합니다.
    for x in (-1,'63'):
        try:print(x,'→',tasks.validate_report_id(x))
        except ValueError as e:print(tasks.classify_error(e))

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
