"""Day21 실습 07: 실패 분류·최대 2회. 실행: python main.py"""
# 전체 흐름
# 1. main()이 미리 작성된 응답 샘플을 읽습니다.
# 2. 네 가지 시나리오에서 run_with_retry()를 실행합니다.
# 3. 응답을 받아 검증하고, 통과하면 즉시 결과를 반환합니다.
# 4. 첫 응답의 JSON 문법·Schema 오류에 한해 수정 요청을 붙여 한 번 더 시도합니다.
# 5. 다른 오류나 두 번째 실패에서는 중단하고, 시도 기록을 파일로 저장합니다.

# Path는 파일 경로를 다루고 sys는 공통 모듈 검색 경로를 설정합니다.
from pathlib import Path
import sys

# 현재 파일이 있는 폴더의 절대 경로입니다.
HERE = Path(__file__).resolve().parent
# parents[0]은 01_basic, parents[1]은 mentee 폴더입니다.
ROLE_ROOT = HERE.parents[1]
# mentee 안의 coursekit 패키지를 가져올 수 있도록 검색 경로에 추가합니다.
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import DATA, read_json, write_json, sha_text, run

# 기준: Day21 PPT 56~63장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.
# 이 함수는 준비된 응답을 순서대로 공급하고 호출 수와 성공 여부를 확인합니다.
from coursekit.retry import run_fixture_scenarios

def repair_prompt(original, error):
    # 완성 D21-07-B01. 오류 이름에 맞는 짧은 수정 안내를 원래 프롬프트에 추가합니다.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: unknown_source는 자동 형식 수정 요청을 만들지 않습니다.
    # BEGIN D21-07-B01
    # 허용한 오류 이름과 안내 문구를 연결한 딕셔너리입니다.
    safe = {
        "json_error":"JSON 문법을 확인하고 코드 블록 없이 JSON 객체 하나만 출력하세요.",
        "schema_error":"필수 네 필드·자료형·길이를 지키세요. source_ids는 양의 정수 목록입니다.",
    }
    # unknown_source 같은 출처 규칙 오류가 들어오면 ValueError로 안내합니다.
    if error not in safe:
        raise ValueError("형식 오류만 수정 요청을 만듭니다")
    # 원래 질문·지식·계약을 보존하고, 줄바꿈 뒤에 오류 종류별 안내를 붙입니다.
    # safe[error]는 위 딕셔너리에서 해당 오류의 문구를 꺼내는 표현입니다.
    repaired = (
        original
        + "\n[수정 요청] 같은 질문·지식·계약을 유지하세요. "
        + safe[error]
        + " 부족한 사실은 만들지 마세요."
    )
    # 이전 원문 전체, traceback, 비밀 값은 붙이지 않습니다.
    # END D21-07-B01
    return repaired

def run_with_retry(generate, prompt, validate):
    # 완성 D21-07-B02. 최초 응답과 수정 응답을 합쳐 최대 두 번 검증합니다.
    # generate는 프롬프트를 받아 응답 원문을 돌려주는 함수입니다.
    # validate는 원문을 받아 (검증된 객체, 오류 이름)을 돌려주는 함수입니다.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 호출 수가 최대 2. 정상 정보 부족도 1회에 종료.
    # 핵심: 성공 반환은 자동 구조·참조 통과일 뿐입니다. 근거를 맞게 읽었는지 사람 검토가 남습니다.
    # BEGIN D21-07-B02
    # 각 시도의 원문과 검사 결과를 차례대로 모읍니다.
    history = []
    # attempt에는 1, 2가 차례대로 들어가므로 호출 횟수가 최대 두 번으로 제한됩니다.
    for attempt in (1,2):
        # 이번 실습의 generate는 미리 작성된 샘플을 하나씩 반환합니다.
        raw = generate(prompt)
        obj, error = validate(raw)
        # sha_text()는 프롬프트의 SHA-256 해시를 만듭니다.
        # 해시는 각 시도에서 사용한 프롬프트가 같은지 비교하는 데 쓸 수 있습니다.
        history.append({"attempt":attempt,"error":error,"raw_text":raw,"prompt_sha256":sha_text(prompt)})
        # 오류가 없으면 즉시 종료합니다. 올바른 정보 부족 응답도 이 경로로 통과합니다.
        if error is None:
            return obj, history
        # 형식 오류 외의 오류이거나 이미 두 번째 시도라면 반복을 끝냅니다.
        if error not in {"json_error", "schema_error"} or attempt == 2:
            break
        # 첫 번째 형식 오류일 때만 다음 시도에 쓸 프롬프트를 수정합니다.
        prompt = repair_prompt(prompt, error)
    # 통과한 응답이 없으면 객체 자리에 None을 돌려주고 실패 기록은 보존합니다.
    return None, history
    # END D21-07-B02

def main():
    # JSON 파일에서 응답 원문과 예상 오류가 들어 있는 샘플 목록을 읽습니다.
    fixtures=read_json(DATA/"validation_fixtures.json")
    # 완성 D21-07-B03. 네 시나리오의 호출 수를 확인하고 결과를 저장합니다.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 복구 성공·연속 실패·규칙 중단·부족 정상 종료 네 경우.
    # BEGIN D21-07-B03
    # runner에 이 파일에서 완성한 함수를 넘겨 실제로 그 함수의 동작을 검사합니다.
    # 복구 성공은 2회, 연속 실패는 2회, 출처 오류 중단은 1회, 정상 정보 부족은 1회입니다.
    results = run_fixture_scenarios(fixtures, runner=run_with_retry)
    # 결과 딕셔너리마다 calls 값만 꺼내 목록으로 만듭니다.
    calls = [result["calls"] for result in results]
    # 예상과 다르면 AssertionError가 발생합니다.
    assert calls == [2, 2, 1, 1]
    # 시나리오별 성공 여부와 전체 시도 기록을 UTF-8 JSON 파일로 저장합니다.
    write_json(HERE/"outputs"/"retry_results.json", {"source_mode":"fixture","results":results})
    # END D21-07-B03
    print("[07] 제어 흐름만 검증했습니다. 실제 모델 복구율은 측정하지 않았습니다.")

if __name__ == "__main__":
    # 파일을 직접 실행하면 공통 run()이 main()을 호출하고 일부 실행 오류를 안내합니다.
    run(main)
