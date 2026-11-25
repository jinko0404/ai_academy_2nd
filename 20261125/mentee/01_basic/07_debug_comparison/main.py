"""실습 07. 검색 결과의 순서를 바꾸기 전후를 비교합니다.

검색 프로그램은 먼저 관련 있어 보이는 자료를 후보로 찾습니다.
Reranking(재정렬)은 그 후보 중 질문과 더 관련 있는 자료를 앞으로 올리는 일입니다.
이 실습에서는 같은 질문에 대해 기존 순서(baseline)와 재정렬한 순서(rerank_only)를 봅니다.

예를 들어 Q07에서 사람이 미리 표시한 정답 자료(Gold)는 두 개입니다.
기존 상위 3개에는 그중 하나가 있고, 재정렬한 상위 3개에는 둘 다 있습니다.
Recall@3은 '상위 3개에서 찾은 Gold 수 ÷ 전체 Gold 수'이므로
기존 점수는 1 ÷ 2 = 0.5, 재정렬한 점수는 2 ÷ 2 = 1.0입니다.
두 점수를 빼서 얼마나 바뀌었는지 보고, 기존 결과에서 어느 단계를 살펴볼지도 기록합니다.

아래 score_delta에서 Q07의 점수 차이 0.5가 어떻게 나오는지 따라가 보세요.
"""
from pathlib import Path  # 파일과 폴더의 위치를 다루는 도구
import argparse  # 터미널에 입력한 --offline 같은 옵션을 읽는 도구
import sys  # 여기서는 파이썬이 코드를 찾는 폴더 목록을 다룰 때 사용

# __file__은 이 main.py의 경로입니다. parents[0]은 실습 폴더,
# parents[1]은 01_basic, parents[2]는 mentee 또는 mentor 폴더입니다.
ROLE_ROOT = Path(__file__).resolve().parents[2]
# 아래 coursekit 코드를 찾을 수 있도록 역할 폴더의 경로를 목록에 넣습니다.
if str(ROLE_ROOT) not in sys.path:
    sys.path.insert(0,str(ROLE_ROOT))

from coursekit.eval_workshops import execute  # 질문별로 두 버전을 비교
from coursekit.eval_report import write_json  # 비교 결과를 JSON 파일로 저장


def score_delta(before: float | None, after: float | None) -> float | None:
    """바꾼 뒤의 점수에서 바꾸기 전의 점수를 뺍니다."""
    # TODO D28-07-B01. before는 기존 점수, after는 바꾼 뒤 점수입니다.
    # Q06처럼 자료에 정답이 없는 질문은 Recall 점수가 None입니다.
    # or는 둘 중 하나라도 조건에 맞으면 참이므로, 한쪽 점수라도 없으면
    # 차이도 None을 돌려줍니다. JSON에서는 null로 보입니다.
    if before is None or after is None:
        return None

    # round(..., 6)은 소수점 아래 여섯 자리까지 반올림합니다.
    # Q07은 1.0 - 0.5 = 0.5입니다. 결과가 양수면 점수가 올랐고,
    # 음수면 내려갔고, 0이면 두 점수가 같습니다.
    return round(after-before,6)

def find_problem_stage(run: dict, gold: list[str]) -> str:
    """실행 기록을 보고 먼저 살펴볼 단계를 이름으로 돌려줍니다."""
    # TODO D28-07-B02. run에는 프로그램 상태와 단계별 자료 ID 목록이 들어 있습니다.
    # diagnose는 공통 코드에 있는 진단 함수입니다. 실행 오류와 Gold가 빈
    # 질문을 먼저 확인한 뒤, 아래 순서로 Gold가 적어도 하나 남았는지 봅니다.
    # 1. candidate_ids(처음 찾은 후보)에서 못 찾으면 candidate_miss입니다.
    # 2. ranked_ids의 앞 2개에서 못 찾으면 ranking_miss입니다.
    # 3. included_ids(AI에게 실제 넘긴 자료)에서 못 찾으면 context_miss입니다.
    # 여기까지 Gold가 있으면 답변 상태를 보고 생성·검사 단계나 답변 근거를 살펴보라고 알려 줍니다.
    from coursekit.eval_experiments import diagnose
    return diagnose(run,gold)


def main():
    # 터미널에 입력한 옵션을 읽고, 두 버전을 비교한 뒤 결과 파일을 저장합니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # 옵션을 생략하면 --offline과 같습니다. 미리 저장된 기록으로 문제 세 개를 비교합니다.
    # --live는 실제 환경에 연결해 Q01 한 문제를 비교합니다. 두 옵션은 함께 쓸 수 없습니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()

    # 함수 이름 뒤에 ()를 붙이지 않으면 함수 자체를 넘깁니다.
    # execute가 두 점수의 차이를 계산할 때와 기존 결과를 진단할 때 이 함수들을 부릅니다.
    hooks={
        "score_delta":score_delta,
        "find_problem_stage":find_problem_stage
    }

    # execute는 점수 두 개와 변화량(delta_recall), 기존 결과의 진단(diagnosis)을 저장합니다.
    # 예를 들어 Q01의 Gold는 기존 순서에서 3위입니다. Recall@3에는 포함되므로
    # 기존 점수는 1.0이지만, 단계 진단은 2위까지만 보기 때문에 ranking_miss가 나옵니다.
    try:
        result=execute(7,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안에서 오류가 나면 여기로 옵니다. 예를 들어 연습 기록을 읽지 못하면
        # 오류 종류를 failure.json에 기록합니다.
        failure={'status':'execution_failed','exercise':7,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 파일에 기록한 뒤 터미널에도 오류를 보여 줍니다.

    # 결과는 main.py 옆 outputs/result.json에 저장됩니다.
    # 질문별 비교 결과는 파일 안의 comparisons 목록에서 볼 수 있습니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 보고서 글을 만드는 실습과 함께 쓰는 코드입니다. 실습 07은 result.json을 열면 됩니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    print('Day28 실습 7 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
if __name__=='__main__':
    main()
