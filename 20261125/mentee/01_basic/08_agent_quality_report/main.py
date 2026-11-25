"""실습 08. 질문마다 검색·답변 결과를 확인하고 한 파일로 정리합니다.

Agent가 이 검색·답변 기능을 도구로 쓸 수 있을지 검토하려면 먼저 결과를
확인할 수 있어야 합니다. 질문 하나씩 검색 점수와 출처를 확인한 뒤,
모든 질문의 결과를 모아 평균과 건수를 계산합니다.

Q01은 회의실 예약 취소 방법을 묻습니다. 연습 기록에서는 정답 자료
FAC-102-C1이 검색 1위에 있어서 Recall@3은 1.0입니다.
Q06은 정확한 과제 마감일을 묻지만, 제공된 자료에는 그 날짜가 없습니다.
이 경우에는 검색 점수를 비워 두고, 프로그램이 답변을 보류했는지 확인합니다.

evaluate_one에서 질문 하나를 평가하고 summarize_all에서 결과를 모으는 순서로 읽어 보세요.
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

from coursekit.eval_workshops import execute  # 질문별 평가와 전체 요약을 실행
from coursekit.eval_report import write_json  # 평가 결과를 JSON 파일로 저장


def evaluate_one(example: dict, run: dict, gold: list[str], grades: dict) -> dict:
    """질문 하나의 검색 점수와 출처 확인 결과를 만듭니다."""
    # TODO D28-08-B01. example은 원래 질문 정보, run은 실행 뒤 받은 기록입니다.
    # gold는 사람이 미리 정한 정답 자료 ID 목록이고, grades는 자료별 관련성 등급입니다.
    # 공통 함수 evaluate_record는 먼저 두 질문 번호가 같은지 확인합니다.
    # 검색 순서, AI에게 보여 준 자료, 답변, 출처 같은 필수 기록이 빠지면 오류가 납니다.
    # 예를 들어 검색 목록 항목이 빠졌다면 실제로 아무것도 못 찾았는지 알 수 없습니다.
    from coursekit.eval_report import evaluate_record

    # k=3은 검색 순위의 앞 3개로 점수를 계산한다는 뜻입니다.
    # 출처 확인 결과(sources_valid)는 답변에 달린 출처가 AI에게 보여 준
    # 자료에 들어 있었는지, 원본 자료의 위치와 맞는지 등을 검사해 기록합니다.
    return evaluate_record(example,run,gold,grades,k=3)

def summarize_all(rows: list[dict]) -> dict:
    """모든 질문의 평가 결과를 모아 평균과 건수를 계산합니다."""
    # TODO D28-08-B02. rows는 evaluate_one에서 만든 질문별 결과의 목록입니다.
    # 공통 함수 report_summary는 실행 방식, 데이터 버전, 검색 설정이 같은지
    # 먼저 확인합니다. 같은 조건의 결과끼리 모아야 평균을 해석할 수 있습니다.
    # Q06처럼 정답 자료가 없는 질문의 Recall은 None입니다. 이 값은 점수 평균에서
    # 빼고, 답변을 제대로 보류했으면 no_answer_correct_abstentions에 1건을 더합니다.
    # 출처 확인에 실패한 건수와 비정상 상태로 끝난 건수도 기록합니다.
    from coursekit.eval_report import report_summary
    return report_summary(rows)


def main():
    # 터미널에 입력한 옵션을 읽고, 평가한 뒤 결과 파일을 저장합니다.
    parser=argparse.ArgumentParser(description="Day28 RAG Evaluation 실습")
    # 옵션을 생략하면 --offline과 같습니다. 미리 저장된 질문 8개의 기록을 사용합니다.
    # --live는 실제 환경에 연결해 질문 목록의 처음 3개를 실행합니다.
    # 두 옵션은 함께 쓸 수 없습니다.
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true',help='고정 학습 데이터만 실행(기본값)')
    mode.add_argument('--live',action='store_true',help='실제 CUDA GPU/DB/LLM 연결. 06~08만 지원하며 API 비용 발생 가능')
    args=parser.parse_args()

    # 함수 이름 뒤에 ()를 붙이지 않으면 함수 자체를 넘깁니다.
    # execute는 질문마다 evaluate_one을 실행하고, 마지막에 summarize_all을 실행합니다.
    hooks={
        "evaluate_one":evaluate_one,
        "summarize_all":summarize_all
    }

    # 검색·답변 프로그램에 보낼 입력은 질문 정보로 만듭니다.
    # 정답 자료는 실행 결과를 받은 뒤 채점 함수에 전달합니다.
    # 저장할 결과에서 questions는 질문별 상세 기록이고, summary는 전체 요약입니다.
    try:
        result=execute(8,hooks,live=args.live,level='01_basic')
    except Exception as exc:
        # try 안에서 오류가 나면 여기로 옵니다. 예를 들어 필수 실행 기록이
        # 빠져 채점할 수 없으면 오류 종류를 failure.json에 기록합니다.
        failure={'status':'execution_failed','exercise':8,
                 'stage':getattr(exc,'stage','evaluation'),
                 'error_type':type(exc).__name__, 'mode':'live' if args.live else 'offline_fixture'}
        write_json(Path(__file__).resolve().parent/'outputs'/'failure.json',failure)
        raise  # 파일에 기록한 뒤 터미널에도 오류를 보여 줍니다.

    # main.py 옆의 outputs 폴더에 평가 결과를 저장합니다.
    output=Path(__file__).resolve().parent/'outputs'/'result.json'
    write_json(output,result)
    # 글로 된 보고서가 결과에 있을 때만 report.md도 씁니다.
    # 기본 실습의 질문별 기록과 전체 요약은 result.json에서 볼 수 있습니다.
    if isinstance(result.get('markdown_report'),str):
        (output.parent/'report.md').write_text(result['markdown_report'],encoding='utf-8')
    print('Day28 실습 8 · 01_basic ·',result['mode'])
    print('출력:',output)
    return result


# python main.py처럼 이 파일을 직접 실행하면 여기서 main()이 시작됩니다.
if __name__=='__main__':
    main()
