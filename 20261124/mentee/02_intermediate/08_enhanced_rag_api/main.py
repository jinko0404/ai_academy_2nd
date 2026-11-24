"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""

# 06번에서 연결한 검색·답변 과정을 이번에는 API로 호출합니다.
# 예를 들어 '회의실 예약을 취소하는 방법은?'을 JSON으로 보내면 답변과 출처를 받습니다.
# 이 파일을 실행하면 FastAPI의 TestClient가 같은 프로그램 안에서 예제 요청을 보내고 응답을 모읍니다.
from pathlib import Path
import sys

# HERE는 실습 폴더입니다. 상위 폴더를 올라가며 coursekit을 찾고, 그 멘티 폴더를
# Python의 모듈 검색 경로에 추가합니다.
HERE=Path(__file__).resolve().parent
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))


import argparse

from coursekit.workshops import STAGE_RUNNERS

from coursekit.io import save_report



from steps import make_service_chain

from steps import wire_fastapi

from checks import check_http_response

def main():
    # 1. --offline을 붙이면 준비된 교육용 데이터로 실행합니다.
    #    기본 실행에서는 실제 E5·PostgreSQL 검색, Cross-Encoder, LLM을 사용합니다.
    parser=argparse.ArgumentParser(description='Day27 실습 08 · 08_enhanced_rag_api')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()

    # 2. STAGE_RUNNERS[8]은 coursekit/workshops.py의 run_08()입니다. 이 함수의
    #    `f1, f2, *more = fns` 줄은 아래 함수 목록을 각각의 변수에 담습니다.
    #    f1 = make_service_chain, f2 = wire_fastapi, more[0] = check_http_response
    # 3. run_08()은 f1로 검색·답변 과정(chain)을 만들고 f2로 FastAPI 앱에 연결합니다.
    #    TestClient가 POST /api/v1/rag/answer로 요청을 보내면 앱이 응답을 돌려줍니다.
    #    more[0]이 응답을 확인합니다. HTTP 상태 코드와 응답 본문, 검사 결과는 rows에 쌓입니다.
    report=STAGE_RUNNERS[8](args.offline,[make_service_chain,wire_fastapi,check_http_response])

    # 4. 모아 둔 응답을 outputs에 저장합니다. 오프라인이면 day27_08_fixture.json,
    #    실제 모델을 쓰면 day27_08_live.json입니다.
    path=save_report(HERE,'day27_08_'+('fixture' if args.offline else 'live')+'.json',report)
    print('실습 08 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# 터미널에서 python main.py로 직접 실행하면 main()을 호출합니다.
if __name__=='__main__':main()
