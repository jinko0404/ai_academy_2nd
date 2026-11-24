"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다."""

# 실습 목적: RAG 다섯 단계의 입력과 출력을 이어 보고, 관련 문서를 찾았더라도
# 질문에 답할 만큼 구체적인 근거가 있는지 별도로 확인하는 연습입니다.
# 전체 흐름: 예시 목록 읽기 → 과제 문서 선택 → RAG 단계 정리와 근거 판단 → JSON 저장.
# 다섯 단계는 개념을 설명하기 위한 목록입니다. 실행 중에는 준비된 예시 데이터만 사용합니다.

from pathlib import Path
import sys

# 현재 파일에서 실습 폴더와 멘티 루트를 구해 터미널 위치가 바뀌어도 같은 지원 모듈을 불러옵니다.
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import save_json, show_success
from coursekit.data import example_hit, load_hits
from coursekit.reporting import execution_record

# 입력·출력 설명은 제공됩니다. TODO에서는 올바른 처리 순서를 연결합니다.
# 단계별 입출력을 이름으로 찾고, 실제 처리 순서는 stage_order()에서 정합니다.
STAGE_CARDS = {
    "Generator": {"input": "모델 입력", "output": "생성된 답변"},
    "Prompt": {"input": "규칙·질문·Context", "output": "모델이 읽을 메시지"},
    "Retriever": {"input": "질문·검색 조건", "output": "관련 청크 후보"},
    "Validator · Source 연결": {"input": "생성 결과·허용 ID", "output": "검증 결과·원문 위치"},
    "Context Builder": {"input": "후보 목록", "output": "선택한 근거·포함 ID"},
}

def stage_order() -> list[str]:
    """STAGE_CARDS의 이름 다섯 개를 처리 순서대로 반환합니다."""
    # BEGIN D26-01-I01
    # TODO D26-01-I01: 다섯 단계의 처리 순서를 반환합니다.
    # 힌트: STAGE_CARDS의 이름을 사용합니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # Retriever가 질문에 맞는 후보 청크를 찾으면 Context Builder가 답변에 쓸 근거와 ID를 고릅니다.
    # Prompt는 규칙·질문·근거를 모델 입력으로 묶고, Generator는 그 입력으로 답변을 만듭니다.
    # 마지막 단계는 답변의 인용 ID를 허용 목록과 대조하고 인용한 근거를 원문 위치에 연결합니다.
    return ["Retriever", "Context Builder", "Prompt", "Generator", "Validator · Source 연결"]
    # END D26-01-I01

def find_hit(hits: list[dict], doc_id: str) -> dict:
    """주어진 예시 목록에서 문서 ID가 같은 Hit 한 건을 선택합니다."""
    # BEGIN D26-01-I02
    # TODO D26-01-I02: 목록에서 요청한 doc_id의 Hit를 반환하고 없으면 LookupError를 발생시킵니다.
    # 힌트: 각 Hit의 doc_id를 순서대로 비교합니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 미리 제공된 검색 결과(Hit)를 순서대로 살펴보고 doc_id가 같은 첫 항목을 반환합니다.
    # Hit 전체를 반환하므로 다음 함수에서 본문과 출처 정보를 함께 읽을 수 있습니다.
    for hit in hits:
        if hit["doc_id"] == doc_id:
            return hit
    
    # 반복을 끝까지 돌았으면 일치하는 문서가 없으므로 조회 실패를 호출자에게 알립니다.
    raise LookupError(f"예시 목록에 문서가 없습니다: {doc_id}")
    # END D26-01-I02

def deadline_note(hit: dict) -> dict:
    """과제 문서의 존재와 정확한 마감 날짜의 존재를 구분해 기록합니다."""
    # BEGIN D26-01-I03
    # TODO D26-01-I03: 문서 발견 여부와 정확한 날짜의 유무를 나누어 기대 판단을 작성합니다.
    # 힌트: “담당자가 공지한 날짜”는 실제 날짜 값이 아닙니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 이 함수는 DOC-ASSIGN 예시 본문을 읽고 사람이 내린 판단을 고정값으로 기록합니다.
    # 인용 ID, 문서 ID, 버전, 원문 위치, 본문을 함께 남겨 판단 근거를 다시 확인할 수 있게 합니다.
    return {
            "question": "내 과제의 정확한 마감 날짜는?",
            "citation_id": hit["citation_id"],
            "doc_id": hit["doc_id"],
            "source_version": hit["source_version"],
            "source_locator": hit["source_locator"],
            "evidence": hit["body"],
            # 과제 문서는 찾았지만 “담당자가 공지한 날짜”라는 안내만으로는 정확한 날짜를 알 수 없습니다.
            # 두 사실을 각각 기록하고 답변에 필요한 근거가 부족하다는 상태를 남깁니다.
            "document_found": True,
            "exact_date_available": False,
            "expected_status": "insufficient_context",
            "reason": "문서는 담당자가 공지한 날짜를 따르라고만 안내하며 정확한 날짜는 제공하지 않습니다.",
    }
    # END D26-01-I03

def main():
    # 공통 데이터인 data/lecture_hits.json을 읽어 마감일 질문에 해당하는 과제 문서를 선택합니다.
    hit = find_hit(load_hits(), "DOC-ASSIGN")
    report = {
        # 개념 리뷰 모드에서는 DB, 임베딩, LLM 실행 여부가 모두 False로 기록됩니다.
        "execution": execution_record("concept_review"),
        # 단계 순서에 맞춰 각 카드의 input·output을 펼치고, 근거 판단은 case에 담습니다.
        "stages": [{"stage": name, **STAGE_CARDS[name]} for name in stage_order()],
        "case": deadline_note(hit),
        "observation": "예시 문서 선택과 사람이 작성한 정보 부족 판단",
    }
    # 이 실습 폴더의 outputs/rag_flow.json에 저장하고 터미널에 결과 경로를 표시합니다.
    path = save_json(HERE, "rag_flow.json", report)
    show_success("01 Intermediate", path)

# tasks.py를 직접 실행할 때도 main.py에서 실행할 때와 같은 흐름을 사용합니다.
if __name__ == "__main__":
    main()
