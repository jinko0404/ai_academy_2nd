"""Day26 실습 7 · 응용. 실행 위치와 무관하게 현재 역할 폴더를 사용합니다."""

# 실습 목적은 Context에 제공한 출처를 기준으로 고정 응답의 형식과 인용 규칙을 검사하는 것입니다.
# 전체 흐름은 예시 문서로 Context 만들기 → 허용 인용 ID 수집 → 응답 사례 검증 → JSON 저장입니다.
# 답변의 숫자, 조건, 예외가 원문과 일치하는지는 검증 결과를 읽은 뒤 사람이 확인합니다.

from pathlib import Path
import sys

# 현재 파일 경로에서 실습 폴더와 멘티 루트를 찾아 이 역할의 coursekit 모듈을 불러옵니다.
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))

from coursekit.io import save_json, show_success
from coursekit.data import example_hit
from coursekit.checkpoints import (
    hits_to_documents, build_context, NO_ANSWER, RagServiceError, receive_bundle,
)
from coursekit.fixtures import fixture_bundle
from coursekit.reporting import execution_record


def common_cases():
    """의도적으로 만든 세 입력입니다. 실제 모델의 응답 로그가 아닙니다."""
    # 정상 답변에는 실제 포함 ID를 넣고, 근거 부족 응답에는 지정 안내문과 부족한 정보 설명을 넣습니다.
    answered = {
        "status": "answered",
        "answer": "일반 기준으로 15분까지는 지각입니다. 사전 승인을 받았다면 운영실에 확인합니다.",
        "source_ids": [201], "missing_information": "",
    }
    insufficient = {
        "status": "insufficient_context", "answer": NO_ANSWER,
        "source_ids": [], "missing_information": "정확한 과제 마감 날짜가 자료에 없습니다.",
    }
    # 정상 답변을 복사한 뒤 인용 ID만 999로 바꿔 허용 집합 검사를 확인합니다.
    unknown = {**answered, "source_ids": [999]}
    # 각 사례를 원본 메시지, 파싱된 응답, 파싱 오류를 담는 검증 입력 묶음으로 만듭니다.
    return [
        {"name": "normal", "bundle": fixture_bundle(answered)},
        {"name": "insufficient", "bundle": fixture_bundle(insufficient)},
        {"name": "unknown_source", "bundle": fixture_bundle(unknown)},
    ]


def allowed_source_ids(included_docs):
    """입력: 실제 Context에 포함된 Document 목록 / 출력: set[int]."""
    # BEGIN D26-07-I01
    # TODO D26-07-I01: included_docs의 metadata에서 citation_id를 모아 set으로 반환합니다.
    # 힌트: Document 목록을 순회하고 doc.metadata["citation_id"]를 읽으세요. 검색 후보 목록은 사용하지 않습니다.
    # 아래 NotImplementedError를 지우고 이 블록을 완성하세요.
    
    # 실제 Context에 포함된 문서의 인용 ID만 집합으로 모읍니다. 문서가 없으면 빈 집합을 반환합니다.
    return {doc.metadata["citation_id"] for doc in included_docs}
    # END D26-07-I01


def inspect_case(case, provided_ids):
    """검증 결과를 JSON으로 기록합니다. 원문 의미의 참/거짓은 자동 판정하지 않습니다."""
    # BEGIN D26-07-I02
    # TODO D26-07-I02: receive_bundle로 검사한 결과를 accepted/rejected 형식으로 기록합니다.
    # 힌트: 성공 시 KnowledgeAnswer.model_dump(), RagServiceError 시 str(exc)를 사용합니다. 반환 Key는 docstring 아래 호출부와 expected를 확인하세요.
    # 아래 NotImplementedError를 지우고 이 블록을 완성하세요.
    # receive_bundle은 묶음의 키, 원본 응답 상태, 파싱 결과, 답변 규칙과 허용 출처를 순서대로 검사합니다.
    try:
        answer = receive_bundle(case["bundle"], provided_ids)
    except RagServiceError as exc:
        # 검증에서 거절되면 실패 지점을 확인할 수 있도록 오류 코드를 기록합니다.
        return {"case": case["name"], "outcome": "rejected", "error": str(exc),
                "result": None, "human_review_required": False}
    # 통과한 객체를 저장용 dict로 변환합니다. 원문과 답변을 사람이 대조하도록 표시합니다.
    return {"case": case["name"], "outcome": "accepted", "error": None,
            "result": answer.model_dump(), "human_review_required": True}
    # END D26-07-I02


def type_and_duplicate_cases():
    """같은 답변의 ID 자료형과 중복 여부만 바꾸어 검사 지점을 비교합니다."""
    # BEGIN D26-07-I03
    # TODO D26-07-I03: 정상 입력에서 ID만 문자열 또는 중복 정수로 바꾼 두 fixture_bundle 사례를 반환합니다.
    # 힌트: case 이름은 string_id와 duplicate_id입니다. 다른 네 필드는 정상 예시와 같게 유지하세요.
    # 아래 NotImplementedError를 지우고 이 블록을 완성하세요.
    data = {"status": "answered", "answer": "15분까지는 지각입니다.",
            "source_ids": [201], "missing_information": ""}
    # 문자열 ID는 WireAnswer 파싱에서 걸리고, 중복 정수 ID는 파싱 뒤 인용 규칙 검사에서 걸립니다.
    return [
        {"name": "string_id", "bundle": fixture_bundle({**data, "source_ids": ["201"]})},
        {"name": "duplicate_id", "bundle": fixture_bundle({**data, "source_ids": [201, 201]})},
    ]
    # END D26-07-I03


def run():
    # 강의 출석 예시 한 건으로 Context를 만들고, 같은 Context의 허용 ID를 모든 사례에 적용합니다.
    documents = hits_to_documents([example_hit()])
    context, included_docs = build_context(documents)
    provided_ids = allowed_source_ids(included_docs)
    cases = common_cases() + type_and_duplicate_cases()
    # 실행 모드와 포함 ID, 다섯 사례의 결과를 함께 남겨 각 판정을 원래 입력과 대조할 수 있게 합니다.
    return {
        "source_mode": "validation_fixtures",
        "execution": execution_record("offline"),
        "context": context, "included_ids": sorted(provided_ids),
        "cases": [inspect_case(case, provided_ids) for case in cases],
        "grounding_review": "reports/review.md에서 숫자·조건·예외와 원문 위치를 직접 대조합니다.",
    }


def main():
    report = run()
    # 결과를 outputs/validation_report.json으로 저장하고, 터미널에 사례별 결과와 오류를 표시합니다.
    path = save_json(HERE, "validation_report.json", report)
    show_success("07 · 고정 입력 검증표 (모델·DB 미실행)", path)
    for row in report["cases"]:
        print(row["case"], row["outcome"], row["error"])
    print("accepted는 기계 검사를 통과했다는 뜻입니다. 원문 대조를 마쳐야 학습이 완료됩니다.")


if __name__ == "__main__":
    main()
