"""Day21 실습 05: 개선용 질문 A/B 비교. 실행: python main.py"""
# 전체 흐름
# 1. 명령어를 읽어 prepare(준비), review(검토), fixtures(샘플 검사) 중 하나를 실행합니다.
# 2. prepare는 A/B 규칙과 질문 3개를 준비해 프롬프트 6개와 기록용 파일을 만듭니다.
# 3. 사용자가 각 프롬프트를 모델 화면에 입력하고 실제 응답 원문을 별도로 등록합니다.
# 4. review는 등록된 응답을 검사하고 make_report()로 집계한 결과를 저장합니다.
# 5. fixtures는 제공된 샘플로 검증·재시도 처리가 어떻게 동작하는지 확인합니다.
# 이 프로그램은 모델에 자동으로 요청을 보내지 않습니다.

# Path는 파일 경로를 다루고, sys는 공통 모듈을 찾을 경로를 설정합니다.
from pathlib import Path
import sys

# 현재 main.py가 있는 폴더를 구합니다. 실행하는 터미널의 위치와 관계없이 사용합니다.
HERE = Path(__file__).resolve().parent
# parents[0]은 01_basic, parents[1]은 mentee입니다.
ROLE_ROOT = HERE.parents[1]
# mentee 안의 coursekit 패키지를 가져올 수 있도록 모듈 검색 경로에 추가합니다.
sys.path.insert(0, str(ROLE_ROOT))
# DATA는 mentee/data 경로입니다. 공통 파일 읽기·저장·실행 함수도 가져옵니다.
from coursekit.io import DATA, read_json, write_json, run
from coursekit.prompting import BASE_RULE, ADDED_RULE, make_bundle

# 기준: Day21 PPT 37~45장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.
from coursekit.experiments import prepare_experiment, review_experiment, summarize, write_summary
from coursekit.retry import run_fixture_scenarios
from coursekit.io import check_name

def choose_bundle(bundle_path=None):
    # bundle은 템플릿, 지식, 예제, 출력 계약, A/B 규칙 등을 담은 딕셔너리입니다.
    # 이 실습에서는 새 비교를 만들므로 외부 bundle 경로를 받으면 오류로 안내합니다.
    if bundle_path is not None:
        raise ValueError("05는 새 A/B를 설계합니다. --bundle은 08에서 사용하세요.")
    # TODO D21-05-B01: A의 규칙을 유지하면서 B에 근거 부족 규칙 한 개만 추가하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 질문·문맥·계약·예제는 같고 B의 근거 규칙 1개만 다릅니다.
    # 핵심: 02의 예제 교체 실험과 05의 근거 규칙 실험을 한 비교로 섞지 않습니다.
    # BEGIN D21-05-B01
    # A는 기본 근거 규칙을 사용합니다.
    rule_a = BASE_RULE
    # B는 A의 문장을 유지하고 줄바꿈 뒤에 '근거 부족 시 처리' 규칙 하나를 추가합니다.
    rule_b = rule_a + "\n" + ADDED_RULE  # TODO: ADDED_RULE 추가
    # 이번 A/B의 예제 유무는 둘 다 동일합니다(여기서는 0개).
    # 예제는 양쪽 모두 빈 목록입니다. 다른 조건을 맞춰 추가한 규칙의 영향을 비교합니다.
    bundle = make_bundle(rule_a=rule_a, rule_b=rule_b, examples=[])
    # END D21-05-B01
    return bundle

def choose_cases():
    # 비교에 사용할 개발용 질문 3개를 불러오는 함수입니다.
    # TODO D21-05-B02: 개선용 D01~D03 세 질문만 선택하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 기본필수 3질문 × A/B = 첫 응답 6개. 예상 행동을 Prompt에 넣지 않습니다.
    # BEGIN D21-05-B02
    # JSON 파일을 파이썬 목록으로 읽습니다. 각 항목은 질문과 선택 지식 ID 등을 담습니다.
    cases = read_json(DATA/"dev_v1.json")
    expected_ids = ["D01", "D02", "D03"]  # TODO: D01, D02, D03
    # TODO: cases의 case_id 목록과 expected_ids 대조. H는 아직 실행하지 않습니다.
    # 각 항목 c에서 case_id만 꺼내 목록으로 만든 뒤 예상 목록과 순서까지 비교합니다.
    # 값이 다르면 AssertionError로 중단됩니다. 'D01'과 'D1'은 서로 다른 문자열입니다.
    assert [c["case_id"] for c in cases]==expected_ids
    # END D21-05-B02
    return cases

def make_report(rows):
    # rows는 응답별 검사 결과 딕셔너리들의 목록입니다.
    # 각 행에는 variant(A/B), attempt(시도 번호), 실행 상태, 검사·평가 결과가 들어 있습니다.
    # TODO D21-05-B03: 첫 응답만 골라 통과·미실행·미평가 개수를 기록하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 재시도 응답은 첫 응답 점수에 섞이지 않습니다. 실패를 분모에서 빼지 않습니다.
    # 핵심: None과 False는 다릅니다. 미평가를 0점으로 대체하지 않습니다.
    # BEGIN D21-05-B03
    # 첫 번째 응답만 모아야 합니다. attempt=1은 첫 응답, attempt=2는 재시도입니다.
    # 미완성 TODO: 현재 if False는 모든 행을 제외해 first가 항상 빈 목록이 됩니다.
    # 조건을 r["attempt"] == 1로 바꾸면 첫 응답만 선택할 수 있습니다.
    first = [r for r in rows if False]  # TODO: attempt == 1
    # A와 B의 집계 결과를 담을 딕셔너리입니다.
    counts = {}
    # 같은 집계를 A에 한 번, B에 한 번 수행합니다.
    for variant in ("A","B"):
        # 미완성 TODO: first에서 현재 variant에 해당하는 행만 골라야 합니다.
        # [r for r in first if r["variant"] == variant]로 선택할 수 있습니다.
        subset = []  # TODO: first에서 이 variant 선택
        # counts["A"], counts["B"]에 각각 결과를 저장합니다.
        # N은 해당 그룹의 전체 첫 응답 수이며, 실패와 미실행도 포함합니다.
        # sum()은 조건이 True일 때 1, False일 때 0으로 더해 해당 항목 수를 셉니다.
        # json_passed는 JSON 검사 통과 수, not_run은 아직 실행하지 않은 수입니다.
        # human_pending은 evidence_ok가 None인 수, 즉 근거에 대한 사람 평가가 없는 수입니다.
        # True는 통과, False는 실패, None은 미평가를 뜻하므로 is로 구분합니다.
        counts[variant] = {
            # TODO: json_ok is True의 수
            "N":len(subset), "json_passed":sum(r["json_ok"] is True for r in subset),
            # TODO: execution_state == 'not_run' 
            "not_run":sum(r["execution_state"]=="not_run" for r in subset),
            # TODO: evidence_ok is None
            "human_pending":sum(r["evidence_ok"] is None for r in subset),  
        }
    # 공통 집계기는 전체 rows를 받아 내부에서 첫 응답과 재시도를 구분해 상세 지표를 만듭니다.
    report = summarize(rows)  # 세부 지표·분모는 제공된 집계기
    # 공통 보고서에 위에서 직접 계산한 A/B별 개수도 추가합니다.
    report["learner_first_counts"] = counts\
    # END D21-05-B03
    return report

def choose_repeats():
    # 질문 하나를 A/B 각각 한 번씩 실행하도록 설정합니다. 3질문 × 2버전 × 1회 = 6개입니다.
    # repeat는 같은 조건의 반복 번호이며, 오류 후 재시도를 뜻하는 attempt와 구분됩니다.
    return 1

def main():
    # argparse는 터미널에서 전달한 명령과 옵션을 읽는 표준 라이브러리입니다.
    import argparse
    parser=argparse.ArgumentParser(description="모델 호출 없이 Prompt 준비 또는 실제 수동 응답 검토")
    # command를 생략하면 prepare로 실행합니다. choices는 입력 가능한 명령을 제한합니다.
    parser.add_argument("command",choices=["prepare","review","fixtures"],nargs="?",default="prepare")
    # --run은 outputs 아래에 만들거나 검토할 실험 폴더 이름입니다.
    parser.add_argument("--run",default="dev_01",help="영문 실험 폴더명; 기존 실험 덮어쓰기 금지")
    # type=Path는 입력 문자열을 경로 객체로 바꿉니다.
    # 이 05 실습의 prepare에서는 choose_bundle()이 --bundle 지정을 오류로 처리합니다.
    parser.add_argument("--bundle",type=Path,help="05에서 고정한 frozen_bundle.json. 미지정 시 제공 독립 시작점 사용")
    args=parser.parse_args()
    # 실험 이름에 허용된 문자·길이인지 확인한 뒤 저장 경로를 만듭니다.
    check_name(args.run)
    folder=HERE/"outputs"/args.run
    if args.command=="prepare":
        # 준비 단계. 규칙 묶음, 질문, 반복 횟수를 모아 파일을 생성합니다.
        bundle=choose_bundle(args.bundle)
        cases=choose_cases()
        repeats=choose_repeats()
        # 프롬프트와 응답 등록용 파일 등을 준비합니다. 실제 응답은 이후 사용자가 등록합니다.
        prepare_experiment(folder,bundle,cases,repeats=repeats)
        print("[준비]",folder)
        print("prompts의 원문을 각각 새 대화에 입력하세요. 아직 실제 응답은 0개입니다.")
    elif args.command=="review":
        # 검토 단계. 등록된 응답과 사람 평가를 읽어 검사 결과 목록을 만듭니다.
        rows=review_experiment(folder)
        # 검사 결과를 보고서로 집계한 뒤 요약 파일로 저장합니다.
        report=make_report(rows)
        write_summary(folder,report)
        print("[검토]",folder/"summary.md")
        print("raw가 없는 항목은 미실행, 사람 평가 null은 미평가입니다.")
    else:
        # 남은 명령은 fixtures입니다. 미리 작성된 샘플로 재시도 시나리오를 확인합니다.
        # 여기서의 호출 횟수는 샘플 처리 과정에서 센 값입니다.
        results=run_fixture_scenarios(read_json(DATA/"validation_fixtures.json"))
        write_json(HERE/"outputs"/"fixture_checks.json",{"source_mode":"fixture","results":results})
        print("[샘플 검사] 호출 수 2·2·1·1. 모델의 실제 복구율이 아닙니다.")

if __name__ == "__main__":
    # 파일을 직접 실행하면 공통 run()이 main()을 호출합니다.
    # run()은 미완성 TODO, 파일 누락, 잘못된 값·키 등 일부 오류를 안내합니다.
    run(main)
