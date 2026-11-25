"""실습 05. AI가 답변에서 한 말을 보여 준 자료로 확인할 수 있는지 살펴봅니다.

Context는 AI가 답할 때 받은 자료입니다. Claim은 답변에서 하나씩 확인할 수
있는 말입니다. 예를 들어 "15분까지는 지각이고 벌금은 5만원이다"에는
지각 기준과 벌금 액수라는 두 가지 주장이 들어 있습니다.

연습 사례 G01의 자료에는 지각과 결석 기준이 적혀 있습니다. 답변 속 주장
세 개 중 두 개는 자료에서 확인되고, "벌금이 5만원"은 확인되지 않습니다.
확인되는 주장 수를 전체 주장 수로 나눈 2 ÷ 3이 근거 충실도(Faithfulness)입니다.

이 실습은 사람이 미리 매겨 둔 supported 값을 읽어 비율을 계산합니다.
아래 첫 함수에서 G01의 2 ÷ 3이 어떻게 나오는지 따라가 보세요.
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

from coursekit.eval_workshops import execute  # 연습 데이터를 읽고 사례별로 계산
from coursekit.eval_report import write_json  # 계산 결과를 JSON 파일로 저장


def claim_faithfulness(claims: list[dict]) -> float | None:
    """자료에서 확인되는 주장 비율을 돌려줍니다. 확인할 주장이 없으면 None입니다."""
    # TODO D28-05-B01. claims는 여러 주장을 담은 목록입니다.
    # 항목 하나는 {'text': '15분까지는 지각', 'supported': True}처럼 생겼습니다.
    # text는 주장 내용이고 supported는 자료에서 확인되는지 표시합니다.
    # G04처럼 claims가 []이면 전체 개수가 0입니다. 이때는 나누기 전에
    # None을 돌려줍니다. JSON 결과에서는 null로 보입니다.
    if not claims:
        return None

    # for c in claims는 주장을 하나씩 꺼냅니다. bool(c['supported'])로
    # 확인 여부를 True 또는 False로 읽고, sum은 True를 1, False를 0으로 더합니다.
    # G01에서는 1+1+0을 전체 주장 수인 len(claims), 즉 3으로 나눕니다.
    return sum(bool(c['supported']) for c in claims)/len(claims)

def describe_claim_result(claims: list[dict]) -> str:
    """근거가 부족한 주장이나 확인 결과를 한 줄로 알려줍니다."""
    # TODO D28-05-B02. G04처럼 확인할 주장이 하나도 없으면 여기서 끝냅니다.
    if not claims:
        return '검토할 Claim 없음: 별도 판정 필요'

    # for c in claims로 주장을 하나씩 봅니다. supported가 False인 경우에만
    # c['text']를 꺼내 새 목록에 담습니다. G01에서는 ['벌금이 5만원']이 됩니다.
    unsupported=[c['text'] for c in claims if not c['supported']]

    # 아래 한 줄은 if unsupported부터 읽으면 됩니다. 목록에 내용이 있으면
    # ', '.join(...)으로 쉼표와 공백을 사이에 넣어 한 줄로 이은 뒤 앞에 안내를 붙입니다.
    # 목록이 비어 있으면 else 뒤의 문장을 돌려줍니다. 예를 들어 G03은
    # 회의실 사용 승인 이야기가 자료에 있어 점수는 1.0입니다. 과제 제출기한을
    # 묻는 질문에 맞는 답인지 확인하려면 질문과 답변을 따로 비교해야 합니다.
    return '근거 불충분: '+', '.join(unsupported) if unsupported else '모든 Claim이 제공된 Context와 일치'


def main():
    # 터미널에 입력한 옵션을 읽고, 사례별로 계산한 뒤 결과를 저장합니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # --offline과 --live는 함께 쓸 수 없습니다. 실습 05는 --offline을 사용합니다.
    # 옵션을 생략해도 같은 연습 데이터를 읽습니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()

    # 함수 이름에 ()를 붙이지 않으면 함수 자체를 넘길 수 있습니다.
    # execute는 data/generation_claims.json의 사례마다 아래 두 함수를 실행합니다.
    hooks={
        "claim_faithfulness":claim_faithfulness,
        "describe_claim_result":describe_claim_result
    }

    # execute는 결과를 generation_reviews 목록에 담습니다. 각 항목에는
    # 사례 번호(case_id), 점수(faithfulness), 이유(reason)가 들어갑니다.
    # label_source의 human_fixture는 사람이 미리 표시한 연습 데이터라는 뜻입니다.
    try:
        result=execute(5,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안에서 오류가 나면 여기로 옵니다. 예를 들어 실습 05에서
        # --live를 쓰면 실행이 멈추고, 오류 종류를 failure.json에 기록합니다.
        failure={'status':'execution_failed','exercise':5,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 파일에 기록한 뒤 터미널에도 오류를 보여 줍니다.

    # main.py 옆의 outputs 폴더에 사례별 점수와 이유를 저장합니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 보고서 글을 만드는 실습과 함께 쓰는 코드입니다. 실습 05는 result.json을 열면 됩니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    print('Day28 실습 5 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
if __name__=='__main__':
    main()
