"""Day21 실습 06: JSON·Schema·참조 검증. 실행: python main.py"""
# 전체 흐름
# 1. main()에서 검사할 샘플 26개를 읽습니다.
# 2. validate_answer()가 샘플 하나의 검증을 진행합니다.
# 3. parse_answer()에서 JSON 문법을 확인한 뒤 필드의 자료형과 길이를 검사합니다.
# 4. check_rules()에서 출처 ID와 응답 상태에 맞는 값의 조합을 검사합니다.
# 5. 실제 오류가 예상 오류와 같은지 확인하고 검사 결과와 JSON Schema를 저장합니다.

# Path는 파일 경로를 다루고, sys는 공통 모듈을 찾을 경로를 설정합니다.
from pathlib import Path
import sys

# 현재 파일이 있는 폴더의 절대 경로입니다.
HERE = Path(__file__).resolve().parent
# parents[0]은 01_basic, parents[1]은 mentee 폴더입니다.
ROLE_ROOT = HERE.parents[1]
# mentee 안의 coursekit 패키지를 가져올 수 있도록 검색 경로에 추가합니다.
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import DATA, read_json, write_json, run

# 기준: Day21 PPT 46~55장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.
# Annotated는 자료형에 검증 조건을 붙이고, Literal은 허용할 값을 지정합니다.
from typing import Annotated, Literal
# BaseModel은 검증 모델의 기반, ConfigDict는 설정, Field는 필드별 조건입니다.
# ValidationError는 Pydantic 검증에 실패했을 때 발생하는 예외입니다.
from pydantic import BaseModel, ConfigDict, Field, ValidationError
# strict_json은 JSON 읽기와 크기·중복 키·비표준 상수 검사를 담당합니다.
# INSUFFICIENT는 정보 부족 상태에서 사용할 고정 답변 문구입니다.

def make_model():
    # 응답이 갖춰야 할 모양과 조건을 정의한 클래스를 만들어 반환합니다.
    # TODO D21-06-B01: PPT 50장의 엄격한 네 필드 계약을 선언하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 문자열 ID "101", bool true, 추가 필드, 501자 답을 거부합니다.
    # 핵심: 모델 정의는 로컬 검증입니다. 실제 LLM의 생성 형식을 자동 제약하는 API 설정은 아닙니다.
    # BEGIN D21-06-B01
    # 출처 ID는 0보다 큰 정수여야 합니다. gt는 greater than(초과)을 뜻합니다.
    # strict=True는 "101" 같은 문자열이나 True 같은 불리언을 정수로 받아들이지 않습니다.
    SourceID = Annotated[int, Field(strict=True, gt=0)]
    class KnowledgeAnswer(BaseModel):
        # extra="forbid"는 정의한 네 필드 외에 다른 필드가 있으면 거부합니다.
        # strict=True는 잘못된 자료형을 자동 변환하지 않고 오류로 처리합니다.
        model_config = ConfigDict(extra="forbid", strict=True)
        # 상태는 아래 두 문자열 중 하나여야 합니다.
        status: Literal["answered", "insufficient_context"]
        # 답변은 1~500자 문자열입니다. 공백만 있는 답변은 뒤의 check_rules()에서 검사합니다.
        answer: str = Field(min_length=1, max_length=500)
        # 출처는 최대 3개의 양의 정수 목록입니다. 빈 목록의 허용 여부는 상태에 따라 검사합니다.
        source_ids: list[SourceID] = Field(max_length=3)
        # 부족한 정보 설명은 최대 200자이며, 빈 문자열도 자료형 검사는 통과합니다.
        missing_information: str = Field(max_length=200)
    # 기본값을 추가하지 마세요. 네 필드 모두 필수입니다.
    # END D21-06-B01
    return KnowledgeAnswer

def parse_answer(raw):
    # raw는 검사할 응답 원문 문자열입니다. Model에는 검증 클래스가 담깁니다.
    Model=make_model()
    # TODO D21-06-B02: JSON 문법 실패와 Pydantic Schema 실패를 분리하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 깨진 JSON→json_error, {} 또는 문자열 출처→schema_error.
    # 핵심: 형변환을 허용해 잘못된 출처를 억지로 정수로 만들지 않습니다.
    # BEGIN D21-06-B02
    try:
        # JSON 문자열을 파이썬 객체로 바꿉니다. 문법 오류와 제공 함수의 보호 조건을 검사합니다.
        data = strict_json(raw)  # 크기·중복 키 보호는 제공 함수
    except (ValueError, RecursionError):
        # 읽기에 실패하면 객체 자리에는 None, 오류 자리에는 오류 이름을 반환합니다.
        return None, "json_error"
    try:
        # 필수 필드, 자료형, 허용 상태, 길이 등을 검사해 검증된 객체를 만듭니다.
        obj = Model.model_validate(data)
    except ValidationError:
        # JSON은 읽었지만 정의한 응답 형식을 만족하지 못한 경우입니다.
        return None, "schema_error"
    # END D21-06-B02
    # 성공하면 (검증된 객체, 오류 없음)을 반환합니다.
    return obj,None

def check_rules(obj, provided_ids):
    # obj는 자료형 검사를 통과한 응답, provided_ids는 이번 질문에 실제 제공한 지식 ID입니다.
    # TODO D21-06-B03: 이번 문맥의 출처 범위와 상태별 조합을 확인하세요.
    # 입력·대상: obj 및 실제 전달된 provided_ids
    # 완료 기준: [999]→unknown_source; 부족 응답은 고정문·빈 출처·부족 정보명 조합.
    # 핵심: ID 포함 검사는 근거 문장의 의미적 뒷받침을 증명하지 않습니다.
    # BEGIN D21-06-B03
    ids = obj.source_ids
    # set()은 중복을 제거합니다. 원래 개수와 다르면 같은 출처 ID가 반복된 것입니다.
    if len(ids) != len(set(ids)):
        return "duplicate_source"
    # 집합의 <=는 부분집합 검사입니다. 응답의 모든 ID가 제공한 ID 안에 있어야 합니다.
    if not set(ids) <= set(provided_ids):
        return "unknown_source"
    if obj.status == "answered":
        # 답변 가능 상태에서는 출처가 하나 이상이고 부족한 정보 설명은 빈 문자열이어야 합니다.
        # 빈 목록은 조건식에서 False이므로 not ids는 '출처가 없다'는 뜻입니다.
        if not ids or obj.missing_information != "":
            return "status_rule"
    else:
        # 정보 부족 상태에서는 출처가 비어 있고 부족한 정보의 이름이 있어야 합니다.
        # strip()은 양끝 공백을 제거하므로 공백만 적은 설명도 잡아냅니다.
        if ids or not obj.missing_information.strip():
            return "status_rule"
    # 길이가 있어도 공백만 들어 있는 답변이면 오류로 처리합니다.
    if not obj.answer.strip():
        return "blank_answer"
    # 정보 부족 답변은 제공된 고정 문구와 정확히 같아야 합니다.
    if obj.status == "insufficient_context" and obj.answer != INSUFFICIENT:
        return "status_rule"
    # END D21-06-B03
    # 모든 규칙을 통과하면 오류 없음(None)을 반환합니다.
    return None

def validate_answer(raw, provided_ids):
    # 먼저 JSON 문법과 응답 형식을 검사합니다.
    obj,error=parse_answer(raw)
    if error:
        # 앞 단계에서 실패하면 바로 반환합니다. 실패한 객체로 다음 검사를 진행하지 않습니다.
        return None,error
    # 앞 검사를 통과한 객체에 출처·상태 조합 규칙을 적용합니다.
    error=check_rules(obj,provided_ids)
    # 규칙 오류가 있으면 (None, 오류), 모두 통과하면 (객체, None)을 반환합니다.
    return (None,error) if error else (obj,None)

def main():
    # fixture는 입력과 예상 결과를 미리 정해 둔 검사 샘플입니다.
    fixtures=read_json(DATA/"validation_fixtures.json")
    # TODO D21-06-B04: 제공 fixture를 분류하고 모델 정의에서 Schema를 내보내세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 26개 fixture가 기대 오류와 일치. semantic_counterexample은 자동 통과하지만 사람 검토 필요.
    # BEGIN D21-06-B04
    # 샘플별 검사 결과를 담을 목록입니다.
    results = []
    for item in fixtures:
        # 샘플의 응답 원문과 제공 출처 ID로 검증 함수를 실행합니다.
        obj, error = validate_answer(item["raw_text"], item["provided_ids"])
        # 실제 오류와 예상 오류가 같으면 matches_expected가 True입니다.
        # 정상 샘플은 양쪽 모두 None이어야 하고, 오류 샘플은 오류 이름이 일치해야 합니다.
        # evidence_ok=None은 근거의 의미적 타당성에 대한 사람 평가가 아직 없다는 뜻입니다.
        results.append({"fixture_id":item["fixture_id"],"source_mode":"fixture",
                        "error":error,"expected_error":item["expected_error"],
                        "matches_expected":error == item["expected_error"],
                        "evidence_ok":None})
    # all()은 모든 항목이 True일 때 True입니다. 하나라도 예상과 다르면 assert가 중단시킵니다.
    assert all(row["matches_expected"] for row in results)
    # 검사 결과 목록을 outputs/fixture_results.json에 저장합니다.
    write_json(HERE/"outputs"/"fixture_results.json", results)
    # 검증 클래스에서 필드·자료형·길이 등의 JSON Schema를 생성해 저장합니다.
    # check_rules()에 작성한 출처 범위와 상태 조합은 별도 함수가 검사합니다.
    write_json(HERE/"outputs"/"answer_schema.json", make_model().model_json_schema())
    # END D21-06-B04
    print("[06] 로컬 검증 완료. 구조·참조 검사와 실제 근거 평가는 다릅니다.")

if __name__ == "__main__":
    # 파일을 직접 실행하면 공통 run()이 main()을 호출합니다.
    # run()은 파일 누락이나 잘못된 값 등 일부 오류를 안내합니다.
    run(main)
