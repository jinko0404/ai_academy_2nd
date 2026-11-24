"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""

# '회의실 예약을 취소하는 방법은?'으로 검색합니다. 그 결과를 관련도 점수로 다시 정렬하고,
# 취소 방법이 담긴 문서가 정렬 전과 후에 각각 상위 두 자리 안에 들어갔는지 비교합니다.
from pathlib import Path
import sys

# HERE는 이 파일이 들어 있는 폴더입니다. 여기서 상위 폴더를 하나씩 올라가며
# coursekit을 찾고, 그 멘티 폴더를 Python의 모듈 검색 경로에 추가합니다.
HERE=Path(__file__).resolve().parent
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))


import argparse

from coursekit.workshops import STAGE_RUNNERS

from coursekit.io import save_report



from steps import compare_rankings

from steps import interpret_change

from checks import check_conditions

def main():
    # 1. --offline을 붙이면 교육용 검색 결과와 점수를 씁니다. 기본 실행은 E5와 PostgreSQL로
    #    후보를 검색하고, GPU에서 Cross-Encoder로 후보별 관련도 점수를 계산합니다.
    parser=argparse.ArgumentParser(description='Day27 실습 07 · 07_before_after')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()

    # 2. STAGE_RUNNERS[7]은 coursekit/workshops.py의 run_07()입니다. 여기서 넘긴 함수 목록은
    #    run_07()의 `f1, f2, *more = fns` 줄에서 아래와 같이 나뉩니다.
    #    f1 = compare_rankings, f2 = interpret_change, more[0] = check_conditions
    # 3. run_07()은 Q01로 후보를 최대 5개 검색한 뒤 같은 후보에 점수를 매겨 다시 정렬합니다.
    #    f1은 정렬 전후의 문서 ID를 비교하고, f2는 그 비교 결과를 문장으로 바꿉니다.
    #    more[0]은 실제 문서 목록을 받아 평가에 쓴 조건을 확인합니다.
    report=STAGE_RUNNERS[7](args.offline,[compare_rankings,interpret_change,check_conditions])

    # 4. 비교 결과, 해석, 조건 검사, 검색·재정렬에 걸린 시간(ms)을 outputs 폴더에 저장합니다.
    #    오프라인이면 day27_07_fixture.json, 실제 검색을 사용하면 day27_07_live.json입니다.
    path=save_report(HERE,'day27_07_'+('fixture' if args.offline else 'live')+'.json',report)
    print('실습 07 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

# 터미널에서 python main.py로 이 파일을 직접 실행하면 main()이 호출됩니다.
if __name__=='__main__':main()
