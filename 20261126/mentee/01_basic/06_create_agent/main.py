"""06 · create_agent + 실제 Tool 2개 · 기본필수 실행 파일.

기본 실행은 예제 질문 준비, Agent 생성, 모델과 도구 실행, 최종 답변 출력 순으로 진행합니다.
--check를 사용하면 모델 API 호출이나 DB 접속 없이 실습 함수 목록을 확인할 수 있습니다.
"""
from pathlib import Path
import sys,argparse

# 멘티 폴더를 모듈 검색 경로에 추가해 공용 coursekit을 불러올 수 있게 합니다.
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# 같은 폴더의 tasks.py에 질문과 Agent를 만드는 함수가 있습니다.
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

    # 보고서 내용과 63번 보고서의 인덱싱 상태를 함께 묻는 질문을 만듭니다.
    question=tasks.question_for_agent(63)
    agent=tasks.make_agent()
    # 질문을 사용자 메시지(role='user')로 전달하는 invoke() 시점에 실제 모델 호출이 시작됩니다.
    # recursion_limit=20은 Agent 실행 그래프에 허용하는 단계 수의 상한입니다.
    result=agent.invoke({'messages':[{'role':'user','content':question}]},config={'recursion_limit':20})
    # messages에는 사용자 질문, 모델 메시지, 실행된 도구 결과가 쌓입니다. 여기서는 마지막 본문만 출력합니다.
    print('[LLM 실제 최종]',result['messages'][-1].content)

# 터미널에서 이 파일을 직접 실행할 때 main()을 호출합니다.
if __name__=='__main__':main()
