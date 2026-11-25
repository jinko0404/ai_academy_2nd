"""실습 03. 검색 결과가 필요한 자료를 얼마나 잘 찾았는지 계산합니다.

Gold는 사람이 질문에 필요하다고 미리 표시해 둔 자료의 ID(이름표)입니다.
ranked는 검색 결과의 ID를 1위부터 차례로 적은 목록입니다.
k는 앞에서 몇 개까지 볼지 정하는 숫자입니다. k가 3이면 1~3위만 봅니다.
이렇게 앞의 k개를 Top-k라고 부릅니다.

예를 들어 Q07에는 필요한 자료가 두 개 있습니다.
고정 연습 결과에서 FAC-102-C1은 2위, FAC-104-C1은 4위에 있습니다.
3위까지만 보면 필요한 자료를 1개 찾았습니다. 이 개수가 hits_at_3에 들어갑니다.
전체 Gold 2개 중 1개를 찾았으므로 Recall@3은 1 ÷ 2 = 0.5(50%)입니다.

아래 calculate_recall에서 Q07의 0.5가 어떻게 나오는지 따라가 보세요.
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


def calculate_recall(ranked: list[str], gold: list[str], k: int) -> float | None:
    """앞에서 k개를 보고 전체 Gold 중 몇 퍼센트를 찾았는지 0~1 사이의 값으로 돌려줍니다."""
    # TODO D28-03-B01. 앞에서 찾은 Gold 개수를 전체 Gold 개수로 나눕니다.
    # k가 정수가 아니거나 1보다 작으면 raise로 여기서 계산을 멈추고 오류를 알려 줍니다.
    if type(k) is not int or k<1:
        raise ValueError('k는 1 이상의 정수')

    # Q06처럼 gold가 []이면 나눌 때 아래쪽 숫자(분모)가 0입니다.
    # 이때는 계산할 점수가 없다는 뜻으로 None을 돌려줍니다. JSON에는 null로 적힙니다.
    # 0.0은 Gold가 있었는데 검색 결과 앞 k개에서 하나도 못 찾았을 때 나옵니다.
    if not gold:
        return None

    # ranked[:k]는 검색 결과의 맨 앞부터 k개를 꺼냅니다.
    # set은 중복을 없애고, &는 두 목록에 모두 있는 ID만 골라냅니다.
    # / 왼쪽 len(...)은 찾은 Gold 개수, 오른쪽 len(...)은 전체 Gold 개수입니다.
    # Q07에서 k=3을 쓰면 왼쪽은 1, 오른쪽은 2이므로 0.5를 돌려줍니다.
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))

def count_gold_hits(ranked: list[str], gold: list[str], k: int) -> int:
    """검색 결과의 앞 k개에 Gold가 몇 개 들어 있는지 정수로 돌려줍니다."""
    # TODO D28-03-B02. 중복 ID는 같은 자료이므로 한 번만 셉니다.
    # 예를 들어 ranked=['A', 'A', 'B', 'C'], gold=['A', 'C'], k=3이라고 해 봅시다.
    # ranked[:3]은 ['A', 'A', 'B']이고 set으로 바꾸면 A와 B만 남습니다.
    # Gold와 겹치는 것은 A 한 개뿐입니다. len(...)이 그 개수인 1을 돌려줍니다.
    return len(set(ranked[:k]) & set(gold))


def main():
    # 터미널에 입력한 옵션을 읽고, 질문별로 계산한 뒤 결과를 저장합니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # --offline과 --live는 함께 쓸 수 없습니다. 실습 03은 --offline을 사용합니다.
    # 옵션을 생략해도 같은 연습 데이터를 읽습니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()

    # 함수 이름 뒤에 ()를 붙이지 않으면 함수 자체를 넘길 수 있습니다.
    # execute가 질문마다 이 두 함수를 불러 개수와 비율을 계산합니다.
    hooks={
        "calculate_recall":calculate_recall,
        "count_gold_hits":count_gold_hits
    }

    # execute는 3번 기본 실습의 Q01, Q07, Q06을 k=3으로 계산합니다.
    # 미리 저장된 검색 결과와 질문별 Gold를 읽어, 각 질문의 결과를 metrics 목록에 넣습니다.
    try:
        result=execute(3,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안에서 오류가 나면 여기로 옵니다. 예를 들어 실습 03에서
        # --live를 쓰면 실행이 멈추고, 오류 종류를 failure.json에 기록합니다.
        failure={'status':'execution_failed','exercise':3,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 파일에 기록한 뒤 터미널에도 오류를 보여 줍니다.

    # main.py 옆의 outputs 폴더에 질문별 계산 결과를 저장합니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 보고서 글을 만드는 실습과 함께 쓰는 코드입니다. 실습 03은 result.json을 열면 됩니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    print('Day28 실습 3 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
if __name__=='__main__':
    main()
