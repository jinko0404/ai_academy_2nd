"""실습 01. AI 답변에 문제가 생겼을 때 어디부터 확인할까요?

예를 들어 "회의실 예약을 어떻게 취소해요?"라는 질문에 답한다고 생각해 보세요.
먼저 관련 문서들을 검색합니다. 이 첫 목록이 검색 후보(candidate)입니다.
그중 몇 개를 골라 AI에게 보여 줍니다. 그 자료가 Context입니다.
AI는 그 자료를 읽고 답합니다. 답에 꼭 필요한 문서인 '예약 취소 안내'처럼
평가자가 미리 정해 둔 정답 근거 문서를 이 실습에서는 Gold라고 부릅니다.

이 파일은 미리 준비한 연습 문제(data/failure_cases.json)를 읽습니다.
문제마다 검색과 답변이 어떻게 됐는지 예(True) 또는 아니요(False)로 적혀 있습니다.
그 값을 읽고 어디에서 문제가 생겼는지 찾아 outputs/result.json에 기록합니다.

처음 읽을 때는 아래 classify_case에서 F01이 어디에서 멈추는지 따라가 보세요.
"""
from pathlib import Path  # 파일이나 폴더의 위치를 다루는 도구
import argparse  # 터미널 명령 뒤에 붙인 --offline 같은 옵션을 읽는 도구
import sys  # 여기서는 파이썬이 코드를 찾는 폴더 목록을 다룰 때 사용

# __file__은 지금 읽고 있는 main.py의 위치입니다.
# parents[0]은 실습 폴더, [1]은 01_basic, [2]는 mentee 또는 mentor 폴더입니다.
ROLE_ROOT = Path(__file__).resolve().parents[2]
# 파이썬이 아래 coursekit 코드를 찾을 수 있도록 역할 폴더를 찾는 목록에 추가합니다.
if str(ROLE_ROOT) not in sys.path:
    sys.path.insert(0,str(ROLE_ROOT))

from coursekit.eval_workshops import execute  # 연습 문제를 읽고 한 건씩 처리
from coursekit.eval_report import write_json  # 결과를 읽어 볼 수 있도록 JSON 파일에 저장


def classify_case(case: dict) -> str:
    """연습 문제 한 건을 받고 처음 문제가 생긴 단계의 이름을 글자로 돌려줍니다."""
    # TODO D28-01-B01. 아래 네 질문을 위에서부터 하나씩 확인합니다.
    # case는 이름과 값으로 정보를 담은 딕셔너리(dict)입니다.
    # 예를 들어 F01은 {'case_id': 'F01', 'system_error': False, ...}처럼 생겼습니다.
    # case.get('system_error')는 그중 system_error에 적힌 값(False)을 꺼냅니다.
    # if는 조건이 맞을 때 아래 줄을 실행합니다. not은 True와 False를 뒤집습니다.
    # return을 만나면 결과를 돌려주고 이 함수를 끝냅니다.

    # 1. 프로그램이 제대로 실행됐나요?
    # F04는 system_error가 True입니다. 검색이 제대로 시작됐는지부터
    # 확인해야 하므로, 여기서 '시스템 오류'라고 답하고 끝냅니다.
    if case.get('system_error'):
        return 'system_failure'

    # 2. 검색으로 처음 찾은 문서에 Gold가 하나라도 있나요?
    # F01은 candidate_has_gold가 False입니다. not을 붙이면 조건이 True가 됩니다.
    # 그래서 '처음 검색에서 필요한 문서를 못 찾음'이라고 답하고 끝냅니다.
    if not case.get('candidate_has_gold'):
        return 'candidate_miss'

    # 3. AI에게 실제로 보여 준 문서에 Gold가 하나라도 있나요?
    # F02는 검색 후보에 예약 취소 안내가 있었지만 보여 줄 문서를 고를 때
    # 빠진 경우와 같습니다. 문서 순서와 어떤 문서를 골랐는지 확인할 차례입니다.
    if not case.get('context_has_gold'):
        return 'ranking_or_context_miss'

    # 4. AI의 답을 보여 준 자료에서 확인할 수 있나요?
    # F03은 필요한 문서를 보여 줬는데도 자료에 없는 말을 답한 경우입니다.
    # answer_grounded가 False로 준비돼 있어 '답변 작성 단계에서 문제'라고 답합니다.
    if not case.get('answer_grounded'):
        return 'generation_failure'

    # F05처럼 위에서 멈추는 조건이 하나도 없으면 이 실습의 네 가지 확인을 통과합니다.
    return 'passed'

def describe_evidence(case: dict, stage: str) -> str:
    """위에서 정한 단계 이름(stage)을 보고, 왜 그렇게 판단했는지 한 줄로 알려줍니다."""
    # TODO D28-01-B02. 아래 딕셔너리에서 단계 이름에 맞는 설명을 찾습니다.
    # 예를 들어 F01의 stage는 'candidate_miss'입니다.
    # evidence['candidate_miss']를 읽으면 검색 후보에 Gold가 없었다는 문장이 나옵니다.
    # case도 함께 전달받지만, 여기서는 이미 계산한 stage만으로 문장을 고를 수 있습니다.
    evidence = {
        'system_failure':'서비스 실행이 실패해 지표를 계산할 수 없음',
        'candidate_miss':'Gold 문서 ID가 최초 검색 후보에 존재하지 않음',
        'ranking_or_context_miss':'Gold는 후보에 있으나 LLM Context에 포함되지 않음',
        'generation_failure':'근거가 Context에 있는데 답변 주장이 지지되지 않음',
        'passed':'검색 후보·Context·답변 검토 조건 충족',
    }
    return evidence[stage]


def main():
    # 명령을 읽고 → 연습 문제를 처리하고 → 결과 파일을 저장하는 함수입니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # 옵션 두 개 중 하나만 적을 수 있습니다. 아무 옵션 없이 실행해도 연습 문제를 사용합니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()  # --live를 적었을 때만 args.live가 True입니다. 실습 01은 --offline을 사용합니다.

    # execute에 어떤 함수를 쓸지 알려 줍니다. 함수 이름 뒤에 ()를 붙이지 않으면
    # 지금 실행하는 대신 함수 자체를 넘겨서 나중에 연습 문제마다 실행할 수 있습니다.
    hooks={
        "classify_case":classify_case,
        "describe_evidence":describe_evidence
    }
    # execute는 1번 기본 실습의 문제를 읽습니다. 각 문제의 단계 이름을 먼저 구한 뒤
    # 그 이름으로 이유를 찾습니다. 결과에는 문제 번호(case_id), 단계(stage), 이유(evidence)가 들어갑니다.
    try:
        result=execute(1,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안의 코드에서 오류가 나면 except 안으로 들어옵니다.
        # 예를 들어 실습 01에서 --live를 주면 실행이 멈춰 failure.json을 씁니다.
        # F04는 시스템 오류를 연습하는 문제라, 분류를 마치면 다른 문제와 함께 result.json에 들어갑니다.
        failure={'status':'execution_failed','exercise':1,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 기록한 뒤에도 오류 내용을 터미널에서 볼 수 있게 합니다.

    # main.py 옆의 outputs 폴더에 결과를 저장합니다. JSON은 이름과 값을 글자로 저장하는 형식입니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 결과에 보고서 글이 있을 때만 report.md도 저장합니다. 이 실습은 result.json을 보면 됩니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    # 터미널에는 사용한 모드와 결과 파일의 위치를 보여 줍니다.
    print('Day28 실습 1 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
# 다른 파일이 이 코드를 가져올 때는 함수만 사용할 수 있습니다.
if __name__=='__main__':
    main()
