"""실습 06. 검색·답변 프로그램에 질문을 보내고 실행 기록을 받아 봅니다.

Q01의 "회의실 예약을 취소하는 방법은?"이라는 질문을 예로 들어 봅시다.
실행할 프로그램(Target)에 질문 번호, 질문 내용, 분야, 이전 대화만 보냅니다.
채점할 때 쓸 정답 자료 ID(Gold)와 모범 답안(Reference Answer)은 평가 쪽에 둡니다.
그래야 프로그램이 질문을 보고 필요한 자료를 찾아가는 과정을 평가할 수 있습니다.

이 실습은 Target을 LangChain의 Runnable로 다룹니다. Runnable은 입력을 주고
실행할 수 있는 객체입니다. target.invoke(request)에서 요청을 보내 실제로 실행합니다.
기본 모드에서는 미리 저장한 연습 기록을 읽습니다. 예를 들어 Q01의 기록에는
순위를 매긴 자료 ID 5개와 AI에 보여 줄 자료로 고른 ID 2개가 담겨 있습니다.

먼저 safe_target_input에서 Q01에 어떤 항목이 남는지 따라가 보세요.
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

from coursekit.eval_workshops import execute  # 질문을 읽고 실습 흐름을 실행
from coursekit.eval_report import write_json  # 실행 기록을 JSON 파일로 저장


def safe_target_input(example: dict) -> dict:
    """원본 질문에서 보낼 항목 네 개를 골라 새 요청을 만듭니다."""
    # TODO D28-06-B01. example은 항목 이름과 값이 짝을 이루는 딕셔너리(dict)입니다.
    # for key in (...)에서 넣을 이름만 하나씩 고르고, example[key]로 해당 값을 꺼냅니다.
    # 예를 들어 key가 'question_id'인 차례에는 'question_id': 'Q01'을 넣습니다.
    # 같은 방식으로 question(질문), category(분야), history(이전 대화)를 복사합니다.
    # 새 딕셔너리에는 지정한 네 항목만 들어가므로 정답 자료나 모범 답안은 빠집니다.
    return {key:example[key] for key in ('question_id','question','category','history')}

def invoke_target(target, request: dict) -> dict:
    """요청을 보내고, 돌아온 기록이 같은 질문의 것인지 확인합니다."""
    # TODO D28-06-B02. 여기서 request는 위 함수가 만든 요청입니다.
    # 실행 직전에 정답을 담는 항목 이름이 요청에 섞였는지 한 번 더 확인합니다.
    # 예를 들어 gold_chunk_ids가 요청에 직접 들어 있으면 여기서 오류가 납니다.
    from coursekit.eval_dataset import assert_no_gold_leakage
    assert_no_gold_leakage(request)

    # 이 줄에서 요청을 넘기고 결과를 받습니다. 미리 만들어 둔 Runnable이
    # 기본 모드에서는 저장된 기록을 읽고, --live에서는 검색·답변 프로그램을 실행합니다.
    result=target.invoke(request)

    # isinstance(result, dict)는 결과가 항목 이름과 값을 담은 딕셔너리인지 확인합니다.
    # 결과가 딕셔너리면 .get('question_id')로 질문 번호를 꺼냅니다. 키가 없으면 None입니다.
    # 두 번호가 같아야 Q01을 보냈을 때 Q01의 기록을 받았는지 알 수 있습니다.
    # 결과 형태가 다르거나 번호가 다르면 raise로 멈추고 오류를 알려 줍니다.
    if not isinstance(result,dict) or result.get('question_id')!=request['question_id']:
        raise ValueError('Target 출력의 질문 ID 불일치')
    return result


def main():
    # 터미널에 입력한 옵션을 읽고, 질문을 처리한 뒤 결과 파일을 저장합니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # --offline과 --live 중 하나를 고릅니다. 옵션을 생략하면 --offline과 같습니다.
    # --live를 고르면 실제 환경에 연결하므로 GPU·DB·모델 설정이 필요할 수 있습니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()

    # 함수 이름 뒤에 ()를 붙이지 않으면 함수 자체를 넘깁니다.
    # execute가 질문마다 먼저 요청을 만들고, 그 요청으로 Target을 실행합니다.
    hooks={
        "safe_target_input":safe_target_input,
        "invoke_target":invoke_target
    }

    # 6번 기본 실습에서는 execute가 일반 함수를 RunnableLambda로 감싸
    # invoke()로 부를 수 있게 만들고, 질문 목록의 처음 세 개를 처리합니다.
    # runner_results의 ranked_ids는 검색 결과를 순서대로 적은 자료 ID 목록입니다.
    # included_ids는 그중 AI에게 보여 줄 자료로 고른 ID이고, source_mode는 기록을 가져온 방식입니다.
    try:
        result=execute(6,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안에서 오류가 나면 여기로 옵니다. 예를 들어 결과의 질문 번호가
        # 다르면 오류 종류를 failure.json에 기록합니다.
        failure={'status':'execution_failed','exercise':6,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 파일에 기록한 뒤 터미널에도 오류를 보여 줍니다.

    # main.py 옆의 outputs 폴더에 실행 기록을 저장합니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 보고서 글을 만드는 실습과 함께 쓰는 코드입니다. 실습 06은 result.json을 열면 됩니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    print('Day28 실습 6 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
if __name__=='__main__':
    main()
