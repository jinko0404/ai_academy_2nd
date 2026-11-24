"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""

# 실습 목적
# 실습 질문은 '그럼 취소는 어떻게 해?'입니다. 앞에서 회의실 예약을 이야기했습니다.
# 이 프로그램은 그 대화를 참고해 검색할 문장을 만들고, 찾은 문서를 바탕으로 답변합니다.
# 이 파일의 main() → steps.py의 두 함수 → checks.py의 validate_trace() 순으로 읽으면 됩니다.
from pathlib import Path
import sys

# 실행 준비. __file__에서 이 파일의 절대 경로를 구한 뒤 parent로 실습 폴더를 찾습니다.
# 상위 폴더를 가까운 순서로 확인해 coursekit이 들어 있는 멘티 폴더를 모듈 검색 경로에 넣습니다.
HERE=Path(__file__).resolve().parent
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))


import argparse

from coursekit.workshops import STAGE_RUNNERS

from coursekit.io import save_report



from steps import prepare_pipeline_request

from steps import wire_langchain_pipeline

from checks import validate_trace

def main():
    # 1. 실행 모드를 정합니다. 명령 뒤에 --offline을 붙이면 교육용 후보·점수·답변을 사용합니다.
    # 기본 모드에서는 E5와 PostgreSQL로 검색하고, Cross-Encoder로 재정렬하며, LLM으로 질문 변환과 답변 생성을 합니다.
    parser=argparse.ArgumentParser(description='Day27 실습 06 · 06_langchain_pipeline')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()

    # 2. STAGE_RUNNERS[6]이 아래 세 함수를 받아 한 번의 실습을 진행합니다.
    #    가. prepare_pipeline_request()로 질문과 앞 대화를 묶어 요청(req)을 만듭니다.
    #    나. wire_langchain_pipeline()으로 검색과 답변 생성에 쓸 함수들을 연결합니다.
    #    다. chain.invoke(req)로 요청을 넣어 실행하고 답변·출처·실행 기록을 받습니다.
    #    라. validate_trace()로 실행 기록을 검사하고 그 결과를 report에 담습니다.
    report=STAGE_RUNNERS[6](args.offline,[prepare_pipeline_request,wire_langchain_pipeline,validate_trace])

    # 3. outputs 폴더에 결과를 JSON으로 저장하고, 터미널에 저장한 위치를 보여 줍니다.
    # 고정 자료를 쓰면 day27_06_fixture.json, 실제 서비스와 모델을 쓰면 day27_06_live.json입니다.
    path=save_report(HERE,'day27_06_'+('fixture' if args.offline else 'live')+'.json',report)
    print('실습 06 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# 터미널에서 python main.py로 실행하면 __name__이 '__main__'이 되어 main()을 호출합니다.
if __name__=='__main__':main()
