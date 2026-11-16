"""Day21 실습 08: AI 지식 답변 생성기 v1. 실행: python main.py"""
# 전체 흐름
# 1. 터미널 명령을 읽고 prepare(준비), review(검토), fixtures(샘플 검사)를 선택합니다.
# 2. prepare는 05의 고정된 A/B 설정과 최종 확인용 H 질문 3개를 연결합니다.
# 3. 질문마다 A/B 프롬프트를 만들어 총 6개의 첫 응답을 받을 준비를 합니다.
# 4. 사용자가 모델 화면에서 실행한 응답 원문을 제공 도구로 등록하고 사람 평가를 작성합니다.
# 5. review는 등록한 응답을 검증하고 실행 상태·자동 검사·사람 평가 결과를 집계합니다.
# 6. fixtures는 미리 작성된 샘플로 검증과 재시도 흐름을 확인합니다.

# Path는 파일 경로, sys는 공통 모듈 검색 경로를 다루는 데 사용합니다.
from pathlib import Path
import sys

# 현재 main.py가 들어 있는 폴더의 절대 경로입니다.
HERE = Path(__file__).resolve().parent
# parents[0]은 01_basic, parents[1]은 mentee 폴더입니다.
ROLE_ROOT = HERE.parents[1]
# mentee 안의 coursekit 패키지를 가져올 수 있도록 검색 경로에 추가합니다.
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import DATA, read_json, write_json, run
# make_bundle()은 지식, 템플릿, 출력 계약, A/B 규칙 등을 하나의 딕셔너리로 묶습니다.
from coursekit.prompting import make_bundle

# 기준: Day21 PPT 64~72장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.
from coursekit.experiments import prepare_experiment, review_experiment, summarize, write_summary
from coursekit.retry import run_fixture_scenarios
from coursekit.io import check_name

def choose_bundle(bundle_path=None):
    # bundle_path는 05에서 저장한 frozen_bundle.json의 경로입니다.
    # bundle은 이번 비교에서 사용할 입력 자료와 규칙을 담은 묶음입니다.
    # TODO D21-08-B01: 05에서 고정한 A/B 입력을 재사용하고 실행에 사용할 버전을 확인하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 05 outputs/dev_01/frozen_bundle.json을 선택적으로 지정. 원문을 바꾸지 않습니다.
    # 핵심: 마지막 프로젝트는 기존 흐름을 다시 연결합니다. 새 API·DB·벡터 검색을 만들지 않습니다.
    # BEGIN D21-08-B01
    if bundle_path is None:
        # 경로를 생략하면 제공 코드로 기본 묶음을 만듭니다.
        bundle = make_bundle()  # 앞 교시 미완성자를 위한 제공 체크포인트
        print("[독립 시작점] 실제 D 실험을 개선한 결과가 아닙니다.")
    else:
        # 경로를 지정하면 05에서 고정한 묶음을 JSON 파일에서 읽어 재사용합니다.
        bundle = read_json(bundle_path)  # TODO: read_json(bundle_path)
    # TODO: context_version, schema_version, A/B 버전의 존재를 확인
    # H 결과를 보고 이 파일을 덮어쓰지 마세요.
    # variants 딕셔너리의 키를 집합으로 바꿔 A와 B 두 버전이 있는지 확인합니다.
    # 현재 검사는 버전 이름 A/B만 확인합니다. 위 TODO의 문맥·Schema 버전 확인은 남아 있습니다.
    assert set(bundle["variants"])=={"A","B"}
    # END D21-08-B01
    return bundle

def choose_cases():
    # holdout은 개발 과정에서 규칙을 조정한 뒤 최종 확인에 사용하는 질문 묶음입니다.
    # TODO D21-08-B02: 최종 확인용 H01~H03만 골라 세 질문의 A/B 계획을 만드세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: A/B 각 3질문, 첫 응답 총 6개. H03의 세 근거를 임의로 줄이지 않습니다.
    # BEGIN D21-08-B02
    # 각 질문에는 case_id, question, selected_ids 등의 항목이 들어 있습니다.
    cases = read_json(DATA/"holdout_v1.json")
    expected_ids = ["H01", "H02", "H03"]  # TODO: H01, H02, H03
    # TODO: 세 질문과 선택 ID를 확인. expected_behavior를 Prompt에 넣지 않습니다.
    # [0], [1], [2]는 각각 첫째, 둘째, 셋째 질문을 선택합니다.
    # 각 질문에 제공할 근거 ID가 정해진 목록과 같은지 확인합니다.
    # H03은 세 근거를 모두 전달해야 하므로 [101, 102, 103] 전체를 검사합니다.
    # expected_ids는 현재 선언만 되어 있습니다. 질문 ID와 순서를 비교하는 검사는 남아 있습니다.
    assert cases[0]["selected_ids"]==[103]
    assert cases[1]["selected_ids"]==[102]
    assert cases[2]["selected_ids"]==[101,102,103]
    # END D21-08-B02
    return cases

def make_report(rows):
    # rows는 응답마다 실행 상태, 시도 번호, 자동 검사 결과, 사람 평가를 담은 목록입니다.
    # TODO D21-08-B03: 검증과 사람 평가를 합치되 미실행·미평가 상태를 그대로 남기세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 정상·부족·형식 실패·참조 실패·사람 평가를 구분하고 B의 승리를 강요하지 않습니다.
    # BEGIN D21-08-B03
    # 공통 집계기가 A/B별 자동 검사와 사람 평가, 재시도 지표를 정리합니다.
    report = summarize(rows)
    # 목록에서 attempt가 1인 행만 고릅니다. 두 번째 수정 응답은 첫 응답 집계에서 제외됩니다.
    first = [r for r in rows if r["attempt"]==1]  # TODO: attempt == 1만 선택
    # 공통 보고서에 프로젝트 전체의 실행 현황을 덧붙입니다.
    # first_target은 첫 응답 대상 수이며, 아직 실행하지 않은 대상도 포함합니다.
    # sum()은 조건이 True인 항목을 1씩 더해 개수를 셉니다.
    # not_run은 실행하지 않은 수, human_pending은 근거에 대한 사람 평가가 없는 수입니다.
    # r.get("evidence_ok")는 키가 없을 때도 None을 반환합니다.
    # None은 미평가 상태이고, 실제 실패로 평가한 False와 구분됩니다.
    # max()는 전체 기록의 가장 큰 시도 번호를 구하며, 기록이 비면 default=0을 사용합니다.
    report["project"] = {
        "name":"AI 지식 답변 생성기 v1", "first_target":len(first),
        "not_run":sum(r["execution_state"]=="not_run" for r in first),# TODO: 미실행 수
        "human_pending":sum(r.get("evidence_ok") is None for r in first),# TODO: evidence_ok가 None인 수
        "max_attempt":max((r["attempt"] for r in rows),default=0),     # TODO: 최대 attempt
        "completion":"실제 관찰과 근거 대조 필요",
    }
    # TODO: max_attempt <= 2 확인. 샘플 통과를 모델 우수성으로 적지 마세요.
    # 최초 응답과 수정 응답을 합쳐 최대 두 번이라는 제한을 확인합니다.
    assert report["project"]["max_attempt"]<=2
    # END D21-08-B03
    return report

def choose_repeats():
    # 질문마다 A/B를 각각 한 번 실행합니다. 3질문 × 2버전 × 1회 = 첫 응답 6개입니다.
    # repeat는 같은 조건의 반복 번호, attempt는 한 응답의 최초·수정 시도 번호입니다.
    return 1

def main():
    # argparse는 터미널에서 전달한 명령과 옵션을 읽는 표준 라이브러리입니다.
    import argparse
    parser=argparse.ArgumentParser(description="모델 호출 없이 Prompt 준비 또는 실제 수동 응답 검토")
    # 명령을 생략하면 prepare를 실행하며, choices 안의 세 명령만 허용합니다.
    parser.add_argument("command",choices=["prepare","review","fixtures"],nargs="?",default="prepare")
    # --run은 outputs 아래에서 사용할 실험 폴더 이름입니다.
    parser.add_argument("--run",default="holdout_01",help="영문 실험 폴더명; 기존 실험 덮어쓰기 금지")
    # --bundle은 재사용할 입력 묶음의 경로입니다. type=Path가 문자열을 경로 객체로 바꿉니다.
    parser.add_argument("--bundle",type=Path,help="05에서 고정한 frozen_bundle.json. 미지정 시 제공 독립 시작점 사용")
    args=parser.parse_args()
    # 실험 이름의 문자·길이를 검사하고 outputs/실험이름 경로를 만듭니다.
    check_name(args.run)
    folder=HERE/"outputs"/args.run
    if args.command=="prepare":
        # 준비 단계. 입력 묶음, 최종 질문, 반복 횟수를 정합니다.
        bundle=choose_bundle(args.bundle)
        cases=choose_cases()
        repeats=choose_repeats()
        # 프롬프트 원문과 응답 등록용 파일 등을 생성합니다.
        # 이후 실제 응답 등록과 사람 평가는 README의 제공 도구 안내에 따라 진행합니다.
        prepare_experiment(folder,bundle,cases,repeats=repeats)
        print("[준비]",folder)
        print("prompts의 원문을 각각 새 대화에 입력하세요. 아직 실제 응답은 0개입니다.")
    elif args.command=="review":
        # 검토 단계. 저장된 응답 원문과 사람 평가를 읽어 검증 결과 목록을 만듭니다.
        rows=review_experiment(folder)
        # 결과를 집계하고 summary.md 등의 요약 파일로 저장합니다.
        report=make_report(rows)
        write_summary(folder,report)
        print("[검토]",folder/"summary.md")
        print("raw가 없는 항목은 미실행, 사람 평가 null은 미평가입니다.")
    else:
        # fixtures 명령은 준비된 샘플로 복구 성공·연속 실패·규칙 중단·정보 부족 종료를 검사합니다.
        # 결과에 source_mode="fixture"를 기록해 샘플 검사라는 출처를 표시합니다.
        results=run_fixture_scenarios(read_json(DATA/"validation_fixtures.json"))
        write_json(HERE/"outputs"/"fixture_checks.json",{"source_mode":"fixture","results":results})
        print("[샘플 검사] 호출 수 2·2·1·1. 모델의 실제 복구율이 아닙니다.")

if __name__ == "__main__":
    # 이 파일을 직접 실행할 때 공통 run()이 main()을 호출합니다.
    # run()은 파일 누락이나 잘못된 값 등 일부 오류를 안내합니다.
    run(main)
