"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""

# 실습 목적
# 검색기가 찾은 후보를 Cross-Encoder가 질문과의 관련도에 따라 다시 정렬합니다.
# 실행 결과에서 재정렬 전후의 문서 ID와 점수를 비교할 수 있습니다.

# 전체 흐름
# 실행 모드 선택 → 후보 검색 → 질문과 본문 짝짓기 → 점수로 재정렬 → 검사 → JSON 저장
from pathlib import Path
import sys

# __file__은 현재 파일의 경로입니다. parent로 이 파일이 들어 있는 실습 폴더를 구합니다.
HERE=Path(__file__).resolve().parent

# 상위 폴더를 가까운 순서로 살펴보고 coursekit 폴더가 있는 멘티용 폴더를 찾습니다.
# 그 경로를 Python의 모듈 검색 목록에 넣어 이 실습의 공통 코드를 불러옵니다.
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))


import argparse

from coursekit.workshops import STAGE_RUNNERS

from coursekit.io import save_report



from steps import create_pairs

from steps import rank_documents

from checks import check_rank_scores

def main():
    # 터미널에서 --offline을 붙이면 준비된 후보와 고정 점수를 사용합니다.
    # 기본 실행은 PostgreSQL과 E5로 후보를 검색하고 GPU에서 Cross-Encoder 점수를 계산합니다.
    parser=argparse.ArgumentParser(description='Day27 실습 05 · 05_cross_encoder_reranking')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()

    # STAGE_RUNNERS[5]는 05번 실습을 진행하는 함수입니다. 먼저 후보를 검색한 뒤,
    # steps.py의 두 함수로 입력 쌍 예시와 재정렬 결과를 만들고 checks.py에서 점수를 검사합니다.
    # report에는 재정렬 전후의 문서 ID, 재정렬 점수, 검사 결과가 들어갑니다.
    report=STAGE_RUNNERS[5](args.offline,[create_pairs,rank_documents,check_rank_scores])

    # 실습 폴더의 outputs 아래에 JSON을 저장합니다. 고정 자료로 실행하면 파일 이름이
    # day27_05_fixture.json이고, 실제 검색으로 실행하면 day27_05_live.json입니다.
    path=save_report(HERE,'day27_05_'+('fixture' if args.offline else 'live')+'.json',report)
    print('실습 05 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# 터미널에서 이 파일을 직접 실행하면 __name__이 '__main__'이 되어 main()이 호출됩니다.
if __name__=='__main__':main()
