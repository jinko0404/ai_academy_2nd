"""05 · PostgreSQL Read-only lookup_status · 기본필수 실행 파일.

기본 실행은 보고서 ID 검사, 기존 DB에서 상태 조회, 출력용 결과 정리 순으로 진행합니다.
--check를 사용하면 DB 접속 없이 실습 함수 목록을 확인할 수 있습니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 같은 폴더의 tasks.py에 ID 검증과 출력 정리 함수가 있습니다.
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

    from coursekit.db import fetch_report_status
    # 문자열 '63'을 검사해 조회에 사용할 정수 ID로 변환합니다.
    rid=tasks.parse_report_id('63')
    # 공용 DB 함수가 읽기 전용 트랜잭션을 열고, ID를 SQL 파라미터로 전달해 SELECT를 실행합니다.
    # reports의 상태와 report_embeddings의 정보를 함께 읽어 현재 벡터의 유효성을 확인합니다.
    raw=fetch_report_status(rid)
    # status_view()는 이미 조회된 raw에서 표시할 값만 꺼냅니다.
    print('[실제 11/20 DB]',tasks.status_view(raw))

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
