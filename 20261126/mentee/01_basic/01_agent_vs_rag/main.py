"""01 · AI Agent vs 고정 RAG/Workflow · 기본필수 실행 파일.

기본 실행은 기존 report_agent_lab DB 확인, 예제 질문의 처리 계획 출력,
63번 보고서의 실제 상태 조회 순으로 진행합니다. --check로 함수 목록을 확인할 수 있습니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 아래에서 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 이 실습 폴더의 tasks.py에 질문 분류와 계획 작성 함수가 있습니다.
import tasks


def main():
    # argparse로 실행 옵션을 읽습니다. 예를 들어 --check가 있으면 args.check가 True입니다.
    ap=argparse.ArgumentParser(description='Day29 · 기존 DB 재사용 / CPU fallback 없음')
    ap.add_argument('--check',action='store_true',help='실제 DB/API 없이 순수 함수 TODO 자가 점검 (실제 실습 아님)')
    ap.add_argument('--serve',action='store_true',help='실습 08 FastAPI 실행(실제 DB·GPU·LLM)')
    args=ap.parse_args()
    if args.check:
        # tasks 모듈에서 이름이 '_'로 시작하지 않고 호출 가능한 항목을 찾아 출력합니다.
        # 이 분기에서 바로 종료하므로 TODO 함수 호출과 DB 조회는 기본 실행에서 이루어집니다.
        print('TODO 함수:',[name for name in dir(tasks) if not name.startswith('_') and callable(getattr(tasks,name))])
        print('실제 연동은 --check 없이 실행해야 합니다.')
        return
    # --serve는 이 파일에서 옵션 값만 읽어 둡니다.
    serve=args.serve

    # 기존 DB에 접속하고 연결된 DB가 report_agent_lab인지 확인합니다.
    from coursekit.db import preflight,fetch_report_status
    print('[기존 DB:',preflight()['db'],']')

    # 예제에는 문서 내용과 인덱싱 상태 요청이 함께 있어 두 도구가 계획에 들어갑니다.
    print(tasks.plan_for_request('RAG 근거 보고서 내용을 찾아 보고서 63번의 인덱싱 상태도 알려줘'))
    # 위 줄은 계획을 출력하고, 이 호출은 SELECT로 실제 63번 보고서의 상태를 읽습니다.
    print('[실제 DB]',fetch_report_status(63))

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
