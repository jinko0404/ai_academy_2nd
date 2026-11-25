"""실습 02. 질문마다 답을 찾을 때 참고해야 할 자료를 확인합니다.

Q01은 "회의실 예약을 취소하는 방법은?"이라는 질문입니다.
취소 방법이 적힌 자료는 FAC-102-C1입니다. 긴 문서를 검색하기 좋게 나눈
조각을 청크(chunk)라고 하고, FAC-102-C1은 그 조각의 이름표(ID)입니다.
이렇게 사람이 질문에 필요한 자료로 미리 골라 둔 ID 목록을 Gold라고 부릅니다.

이 실습은 data/gold_labels.json에서 질문별 Gold를 꺼냅니다.
그다음 각 ID가 실제 자료 모음(data/corpus.json)에 있는지 확인합니다.
Gold는 검색 결과를 나중에 채점할 때 씁니다. 검색할 때는 질문을 바탕으로
자료를 찾아야 하므로, 검색 프로그램에 보낼 입력 예시에는 Gold를 빼 둡니다.

먼저 아래 select_gold에서 Q01의 ID를 어떻게 꺼내는지 읽어 보세요.
"""
from pathlib import Path  # 파일과 폴더의 위치를 다루는 도구
import argparse  # 터미널에서 입력한 --offline 같은 옵션을 읽는 도구
import sys  # 여기서는 파이썬이 코드를 찾는 폴더 목록을 다룰 때 사용

# __file__은 이 main.py의 경로입니다. parents[0]은 실습 폴더,
# parents[1]은 01_basic, parents[2]는 mentee 또는 mentor 폴더입니다.
ROLE_ROOT = Path(__file__).resolve().parents[2]
# 아래 coursekit 코드를 찾을 수 있도록 역할 폴더의 경로를 목록에 넣습니다.
if str(ROLE_ROOT) not in sys.path:
    sys.path.insert(0,str(ROLE_ROOT))

from coursekit.eval_workshops import execute  # 실습에 필요한 파일을 읽고 질문별로 처리
from coursekit.eval_report import write_json  # 결과를 JSON 파일로 저장


def select_gold(question_id: str, gold: dict) -> list[str]:
    """질문 번호를 받아, 그 질문에 필요한 자료의 ID를 중복 없이 돌려줍니다."""
    # TODO D28-02-B01. gold는 질문 번호와 자료 ID 목록을 짝지어 둔 딕셔너리(dict)입니다.
    # question_id가 'Q01'이면 gold[question_id]는 ['FAC-102-C1']입니다.
    # list(...)로 복사해서 이 함수에서 쓸 목록을 만듭니다.
    ids=list(gold[question_id])

    # dict.fromkeys는 목록의 값을 딕셔너리의 이름표(키)로 만듭니다.
    # 같은 키는 하나만 남습니다. list(...)로 키를 다시 목록으로 꺼냅니다.
    # 예: ['A', 'B', 'A'] → ['A', 'B']. 처음 나온 순서가 유지됩니다.
    return list(dict.fromkeys(ids))

def validate_gold_ids(ids: list[str], corpus: dict) -> bool:
    """같은 ID가 두 번 나오거나 자료 모음에 없는 ID가 있으면 False를 돌려줍니다."""
    # TODO D28-02-B02. 아래 한 줄은 and를 기준으로 두 부분으로 읽으면 됩니다.
    # 왼쪽 len(ids)==len(set(ids))는 중복이 있는지 확인합니다.
    # set은 같은 값을 하나만 남깁니다. ['A', 'A']는 길이가 2이고 {'A'}는
    # 길이가 1이므로 두 길이가 다르면 중복이 있다는 뜻입니다.
    # 오른쪽 all(cid in corpus for cid in ids)는 ID를 하나씩 꺼내 자료 모음에서 찾습니다.
    # corpus에는 ID를 키로 자료가 들어 있습니다. 목록의 모든 ID가 있어야 all(...)이 True입니다.
    # and로 이어져 있으므로 두 확인을 모두 통과해야 True를 돌려줍니다.
    # Q06의 Gold는 []입니다. 정확한 과제 마감일을 알려 주는 자료가 없다는 뜻입니다.
    # 빈 목록에는 중복이나 등록되지 않은 ID가 없으므로 이 검사 결과는 True입니다.
    return len(ids)==len(set(ids)) and all(cid in corpus for cid in ids)


def main():
    # 터미널에서 받은 옵션을 읽고, 질문을 처리한 뒤 결과 파일을 저장합니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # --offline과 --live는 함께 쓸 수 없습니다. 실습 02는 --offline을 사용합니다.
    # 옵션을 생략해도 같은 연습 데이터를 읽습니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()

    # execute가 문제마다 사용할 함수 두 개를 넘깁니다. 이름 뒤에 ()를 붙이면
    # 그 자리에서 실행되고, 여기처럼 이름만 적으면 나중에 실행할 수 있습니다.
    hooks={
        "select_gold":select_gold,
        "validate_gold_ids":validate_gold_ids
    }

    # execute는 2번 기본 실습의 질문 Q01, Q05, Q06, Q07을 처리합니다.
    # 질문마다 Gold ID를 꺼내고 검사한 뒤 annotations 목록에 넣습니다.
    # 각 항목의 question_id는 질문 번호, gold_chunk_ids는 Gold ID 목록,
    # valid_ids는 검사 결과입니다. target_example은 검색에 보낼 Q01 입력 예시입니다.
    try:
        result=execute(2,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안에서 오류가 나면 여기로 옵니다. 예를 들어 실습 02에서
        # --live를 쓰면 실행이 멈추고, 오류 종류를 failure.json에 기록합니다.
        failure={'status':'execution_failed','exercise':2,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 파일에 기록한 뒤 터미널에도 오류를 보여 줍니다.

    # main.py 옆의 outputs 폴더에 결과를 저장합니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 보고서 글을 만드는 실습과 함께 쓰는 코드입니다. 실습 02는 result.json을 열면 됩니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    print('Day28 실습 2 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
if __name__=='__main__':
    main()
