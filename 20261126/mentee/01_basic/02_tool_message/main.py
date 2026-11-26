"""02 · AIMessage.tool_calls / ToolMessage · 기본필수 실행 파일.

Tool Calling에서는 모델의 요청을 받은 Python 코드가 도구를 실행하고 결과 메시지를 만듭니다.
이 예제의 실행 순서는 63번 보고서 DB 조회, 예제 요청 생성, 요청 ID를 붙인 결과 출력입니다.
--check를 사용하면 DB 접속 없이 함수 목록을 확인할 수 있습니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 같은 폴더의 tasks.py에 요청과 결과 메시지를 만드는 함수가 있습니다.
import tasks


def main():
    # argparse가 터미널에서 전달된 실행 옵션을 읽습니다.
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

    # 기존 report_agent_lab DB에서 63번 보고서의 상태를 SELECT로 읽습니다.
    from coursekit.db import fetch_report_status
    raw=fetch_report_status(63)

    # 같은 보고서 ID로 모델이 보낼 법한 lookup_status 요청을 코드에서 직접 만듭니다.
    call=tasks.make_request('lookup_status',{'report_id':63},'call_001')
    # DB에서 읽은 원본과 호출 ID가 포함된 결과 메시지를 차례로 출력합니다.
    print('[실제 DB 조회]',raw)
    print('[ToolMessage]',tasks.match_result(call,raw))

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
