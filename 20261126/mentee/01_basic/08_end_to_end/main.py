"""08 · FastAPI → Agent → Tool → 응답 · 기본필수 실행 파일.

기본 실행은 예제 질문 하나로 Agent를 실행하고 결과를 터미널에 출력합니다.
--serve는 로컬 FastAPI 서버를 시작하고, --check는 실습 함수 목록을 확인합니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 같은 폴더의 tasks.py에서 예제 질문과 응답 검증 함수를 가져옵니다.
import tasks


def main():
    # argparse로 터미널에서 전달된 실행 옵션을 읽습니다.
    ap=argparse.ArgumentParser(description='Day29 · 기존 DB 재사용 / CPU fallback 없음')
    ap.add_argument('--check',action='store_true',help='실제 DB/API 없이 순수 함수 TODO 자가 점검 (실제 실습 아님)')
    ap.add_argument('--serve',action='store_true',help='실습 08 FastAPI 실행(실제 DB·GPU·LLM)')
    args=ap.parse_args()
    if args.check:
        # 함수 목록을 출력하고 종료합니다. 이 분기는 Agent 실행과 서버 시작보다 먼저 처리됩니다.
        print('TODO 함수:',[name for name in dir(tasks) if not name.startswith('_') and callable(getattr(tasks,name))])
        print('실제 연동은 --check 없이 실행해야 합니다.')
        return
    serve=args.serve
    import starter
    question=tasks.default_question()
    if serve:
        # 서버를 열기 전에 CUDA 장치와 기존 DB 연결 상태를 확인합니다.
        from coursekit.gpu import require_cuda
        from coursekit.db import preflight
        require_cuda();print('[기존 DB]',preflight())
        import uvicorn
        # starter의 앱에 응답 검증 함수를 넘기고, 이 컴퓨터에서 127.0.0.1:8008로 접속하게 합니다.
        uvicorn.run(starter.make_api(tasks.api_response),host='127.0.0.1',port=8008)
    else:
        # CLI에서도 API 경로와 같은 run_agent()와 api_response()를 사용합니다.
        from coursekit.agent import run_agent
        print('[실제 Agent]',tasks.api_response(run_agent(question)))

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
