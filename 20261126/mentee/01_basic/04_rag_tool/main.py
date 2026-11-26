"""04 · 기존 RAG를 rag_search Tool로 · 기본필수 실행 파일.

기본 실행은 기존 DB 상태 확인, 검색 입력 준비, RAG 실행, 결과 출력 순으로 진행합니다.
--check를 사용하면 DB·GPU 연결 없이 실습 함수 목록을 확인할 수 있습니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 같은 폴더의 tasks.py에 검색 입력 준비와 RAG 실행 함수가 있습니다.
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

    # 기존 DB에 접속해 pgvector 확장과 사용 가능한 보고서 벡터 수를 확인합니다.
    from coursekit.db import preflight
    print('[기존 DB]',preflight())

    # 예제 질문의 검색 범위를 연구계획(RESEARCH_PLAN) 보고서로 제한합니다.
    q=tasks.prepare_search('RAG 근거 인용 정확도 향상 계획은 무엇인가요?','RESEARCH_PLAN')
    result=tasks.invoke_rag(q)
    # 화면에는 처리 상태와 답변의 앞 300자까지 표시합니다. 출처 목록은 result['sources']에 들어 있습니다.
    print('[실제 CUDA E5·Reranker·LLM 결과]',result['status'],result['answer'][:300])

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
