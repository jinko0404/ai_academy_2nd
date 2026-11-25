"""실습 04. 답에 필요한 자료가 검색 결과의 몇 위에 처음 나오는지 확인합니다.

Gold는 사람이 질문에 필요하다고 미리 표시해 둔 자료의 ID(이름표)입니다.
실습 03에서는 상위 3개에 Gold가 몇 개 있는지 셌습니다.
이번에는 1위부터 살펴보고 Gold가 처음 나온 순위를 찾습니다.

예를 들어 Q07은 필요한 자료 FAC-102-C1이 2위에 처음 나옵니다.
1을 그 순위로 나눈 값인 1 ÷ 2 = 0.5가 RR 점수입니다.
먼저 찾을수록 점수가 높습니다. 1위면 1.0이고 3위면 약 0.333입니다.
여러 질문의 RR 점수를 평균 내면 MRR입니다.

아래 첫 함수에서 Q07의 순위가 2가 되는 과정을 따라가 보세요.
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

from coursekit.eval_workshops import execute  # 연습 데이터를 읽고 질문별로 계산
from coursekit.eval_report import write_json  # 계산 결과를 JSON 파일로 저장


def first_relevant_rank(ranked: list[str], gold: list[str], k: int) -> int | None:
    """검색 결과의 앞 k개에서 처음 찾은 Gold의 순위를 돌려줍니다."""
    # TODO D28-04-B01. int | None은 순위 숫자 또는 None을 돌려준다는 뜻입니다.
    # ranked[:k]는 1위부터 k개를 꺼냅니다. k가 3이면 앞의 3개를 봅니다.
    # enumerate(..., 1)은 각 항목에 1부터 번호를 붙입니다.
    # 한 번 돌 때마다 pos에는 순위가, cid에는 그 순위의 자료 ID가 들어갑니다.
    for pos,cid in enumerate(ranked[:k],1):
        # cid in gold는 지금 보는 자료가 Gold 목록에 있는지 묻습니다.
        # Q07은 1위를 지나 2위에서 Gold를 찾습니다.
        # return pos가 2를 돌려주고 함수를 끝냅니다.
        if cid in gold:
            return pos

    # 이 줄까지 왔으면 앞 k개를 다 봤지만 Gold를 찾지 못했다는 뜻입니다.
    return None

def calculate_rr(position: int | None) -> float:
    """첫 Gold 순위를 받아 RR 점수를 돌려줍니다."""
    # TODO D28-04-B02. 아래 한 줄은 if를 먼저 읽으면 이해하기 쉽습니다.
    # position is not None은 순위 숫자가 있다는 뜻입니다. 예를 들어 2이면
    # 1.0 / 2를 계산해 0.5를 돌려줍니다. None이면 else 뒤의 0.0을 돌려줍니다.
    # 이 실습에서 Q06처럼 Gold 목록이 빈 질문은 execute가 이 함수를 건너뛰고
    # 점수를 None으로 남깁니다. 그런 질문은 MRR을 계산할 때 빼고 평균을 냅니다.
    return 1.0/position if position is not None else 0.0


def main():
    # 터미널에 입력한 옵션을 읽고, 질문별로 계산한 뒤 결과를 저장합니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # --offline과 --live는 함께 쓸 수 없습니다. 실습 04는 --offline을 사용합니다.
    # 옵션을 생략해도 같은 연습 데이터를 읽습니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()

    # 함수 이름 뒤에 ()를 붙이지 않으면 함수 자체를 넘길 수 있습니다.
    # execute는 질문마다 첫 Gold 순위를 찾고, Gold가 있는 질문의 RR을 계산합니다.
    hooks={
        "first_relevant_rank":first_relevant_rank,
        "calculate_rr":calculate_rr
    }

    # execute는 4번 기본 실습의 Q01, Q07, Q06에 대해 상위 3개를 봅니다.
    # ranking에 첫 순위(first_rank)와 점수(rr_at_3)를 담습니다.
    # mrr_at_3은 점수가 있는 Q01과 Q07의 평균으로 (1/3 + 1/2) ÷ 2 ≈ 0.417입니다.
    try:
        result=execute(4,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안에서 오류가 나면 여기로 옵니다. 예를 들어 실습 04에서
        # --live를 쓰면 실행이 멈추고, 오류 종류를 failure.json에 기록합니다.
        failure={'status':'execution_failed','exercise':4,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 파일에 기록한 뒤 터미널에도 오류를 보여 줍니다.

    # main.py 옆의 outputs 폴더에 결과를 저장합니다. JSON에서는 None이 null로 보입니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 보고서 글을 만드는 실습과 함께 쓰는 코드입니다. 실습 04는 result.json을 열면 됩니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    print('Day28 실습 4 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
if __name__=='__main__':
    main()
