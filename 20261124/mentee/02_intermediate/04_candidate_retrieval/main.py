"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""

# 실습 목적
# 후보 문서 검색을 실행하고, 검색 조건과 후보 문서 ID, 메타데이터 검사 결과를
# JSON으로 남겨 실습 결과를 다시 확인할 수 있게 합니다.

# 전체 흐름
# 실행 옵션 읽기 → 검색 요청 만들기 → 후보 문서 가져오기 → 메타데이터 검사 → 결과 저장
from pathlib import Path
import sys

# __file__은 이 파일의 경로입니다. 절대 경로로 바꾼 뒤 이 파일이 들어 있는 폴더를 구합니다.
HERE=Path(__file__).resolve().parent

# 상위 폴더를 가까운 순서대로 살펴보고 coursekit 폴더가 있는 곳(mentee)을 찾습니다.
# 그 경로를 모듈 검색 목록 앞에 넣어 아래에서 coursekit을 가져올 수 있게 합니다.
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))


import argparse

from coursekit.workshops import STAGE_RUNNERS

from coursekit.io import save_report



from steps import make_candidate_request

from steps import fetch_documents

from checks import metadata_check

def main():
    # argparse는 터미널에서 파일 이름 뒤에 입력한 실행 옵션을 읽습니다.
    # --offline을 붙이면 args.offline이 True가 되어 교육용 고정 자료로 검색합니다.
    # 기본값은 False이며, 이때는 실제 E5 모델과 PostgreSQL의 pgvector로 검색합니다.
    parser=argparse.ArgumentParser(description='Day27 실습 04 · 04_candidate_retrieval')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()

    # STAGE_RUNNERS[4]는 04번 실습을 진행하는 함수입니다. 실행 모드와 세 함수를 넘기면
    # 요청 생성, 후보 조회, 메타데이터 검사를 차례로 수행합니다.
    # 그 결과인 report에는 실행 모드, 후보 문서 ID, 검색 조건, 검사 결과가 담깁니다.
    report=STAGE_RUNNERS[4](args.offline,[make_candidate_request,fetch_documents,metadata_check])

    # save_report()는 이 실습 폴더의 outputs에 JSON을 저장한 뒤 파일 경로를 반환합니다.
    # 파일 이름 끝의 fixture 또는 live로 어떤 실행 모드에서 나온 결과인지 구분합니다.
    path=save_report(HERE,'day27_04_'+('fixture' if args.offline else 'live')+'.json',report)
    print('실습 04 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# 터미널에서 이 파일을 직접 실행하면 __name__이 '__main__'이 되어 main()이 호출됩니다.
# 다른 파일에서 import하면 __name__에는 모듈 이름인 'main'이 들어갑니다.
if __name__=='__main__':main()
