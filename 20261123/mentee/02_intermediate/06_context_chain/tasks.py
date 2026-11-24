"""실습 06 · Context·출처를 보관하고 Runnable Chain을 실행합니다."""

# 실습 목적은 Context에 실제로 넣은 문서와 인용 ID를 보존하고,
# Prompt와 구조화 출력 모델을 연결해 반환되는 응답 묶음(bundle)을 관찰하는 것입니다.
# 전체 흐름은 예시 Hit 읽기 → Document 변환 → Context·출처 준비 → 체인 실행 → JSON 저장입니다.

from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent
# 어느 폴더에서 실행해도 이 역할 폴더의 coursekit을 사용합니다.
sys.path.insert(0, str(HERE.parents[1]))

# coursekit은 실습용 데이터, 앞 장의 공통 함수, 모델 준비와 결과 기록을 제공하는 프로젝트 내부 패키지입니다.
from coursekit.data import load_hits
from coursekit.checkpoints import hits_to_documents, build_context, make_prompt, WireAnswer
from coursekit.llm import create_chat_model
from coursekit.fixtures import fixture_generation
from coursekit.reporting import execution_record, bundle_record
from coursekit.io import save_json, show_success


def prepare_context(documents, question: str, max_items=2, max_chars=600) -> dict:
    """Document 목록 → Context·포함 문서·포함 ID·Prompt 입력. 원문은 바꾸지 않습니다."""
    # BEGIN D26-06-I01
    # TODO D26-06-I01: build_context로 Context와 포함 문서를 얻고, 포함 ID 및 Prompt 입력을 함께 반환합니다.
    # 힌트: 반환 Key는 context, included_docs, provided_ids, inputs입니다. ID는 included_docs에서만 모읍니다.
    # 완료 후 아래 NotImplementedError를 제거하고 반환값을 확인하세요.
    # build_context는 청크 ID로 중복을 걸러 내고, 문서 개수와 전체 문자 수 상한에 맞는 블록만 담습니다.
    # 문자 수에는 본문과 함께 인용 ID, 원문 위치, 줄바꿈도 포함됩니다.
    context, included_docs = build_context(documents, max_items=max_items, max_chars=max_chars)
    # Context에 실제로 들어간 문서에서만 인용 ID를 모아 모델에 제공한 근거와 연결합니다.
    provided_ids = [doc.metadata["citation_id"] for doc in included_docs]
    
    return {
        "context" : context,
        "included_docs" : included_docs,
        "provided_ids" : provided_ids,
        "inputs" : {
            "question" : question,
            "context" : context
        }
    }
    # END D26-06-I01


def run_generation(prompt, structured_model, inputs: dict):
    """Prompt → Structured Model의 bundle. bundle 검증·승인은 다음 실습입니다."""
    # BEGIN D26-06-I02
    # TODO D26-06-I02: Prompt와 Structured Model을 |로 연결하고 inputs로 실행하여 bundle을 반환합니다.
    # 힌트: Chain의 앞 단계 출력이 뒤 단계 입력입니다. 반환 bundle의 raw·parsed·parsing_error를 그대로 보존합니다.
    # 완료 후 아래 NotImplementedError를 제거하고 반환값을 확인하세요.
    # |로 연결하면 Prompt가 만든 ChatPromptValue가 구조화 출력 모델의 입력으로 전달됩니다.
    generation = prompt | structured_model
    
    # invoke()가 체인을 실행합니다. bundle에는 원본 응답, WireAnswer 파싱 결과, 파싱 오류가 담깁니다.
    return generation.invoke(inputs)
    # END D26-06-I02


def compare_contexts(documents) -> dict:
    """같은 목록에서 중복과 선택 개수를 각각 바꿉니다. LLM 호출은 없습니다."""
    # BEGIN D26-06-I03
    # TODO D26-06-I03: 기본·중복 청크 추가·max_items=1 조건을 각각 실행하여 포함 ID를 비교합니다.
    # 힌트: 세 조건에서 한 가지 설정만 바꿉니다. 모델은 호출하지 않고 build_context 결과만 비교합니다.
    # 완료 후 아래 NotImplementedError를 제거하고 반환값을 확인하세요.
    # 첫 문서가 있다면 같은 청크가 연달아 들어간 새 목록을 만들어 중복 제거를 관찰합니다.
    with_duplicate = [documents[0], documents[0], *documents[1:]] if documents else []
    # 문자 상한은 세 번 모두 같게 유지합니다. 중복과 개수의 효과를 분리해 봅니다.
    _, baseline = build_context(documents, max_items=2, max_chars=600)
    _, duplicated = build_context(with_duplicate, max_items=2, max_chars=600)
    _, one_document = build_context(documents, max_items=1, max_chars=600)
    # 전체 본문 대신 포함 ID를 나란히 기록하여 어떤 문서가 선택되었는지 확인합니다.
    return {
        "baseline_ids": [doc.metadata["citation_id"] for doc in baseline],
        "duplicate_ids": [doc.metadata["citation_id"] for doc in duplicated],
        "one_document_ids": [doc.metadata["citation_id"] for doc in one_document],
        "llm_calls": 0,
    }
    # END D26-06-I03


def main():
    # --offline 옵션으로 교육용 고정 응답을 선택할 수 있습니다. 기본 실행은 실제 모델을 사용합니다.
    parser = argparse.ArgumentParser(description="Context → Prompt → Structured Model 관찰")
    parser.add_argument("--offline", action="store_true", help="고정 bundle 예시만 사용; 실제 모델 미실행")
    args = parser.parse_args()
    # 앞 실습의 미완성 코드에 의존하지 않는 제공된 Document 변환 Checkpoint입니다.
    academic_hits = [hit for hit in load_hits() if hit["category"] == "학사"]
    documents = hits_to_documents(academic_hits)
    question = "15분 늦으면 어떻게 기록되나요?"
    prepared = prepare_context(documents, question)
    experiments = {}
    experiments["context_comparison"] = compare_contexts(documents)
    # 답변 규칙과 질문·Context 자리를 담은 프롬프트 템플릿을 불러옵니다.
    prompt = make_prompt()
    if args.offline:
        # 이 Runnable은 제공된 고정 bundle을 반환합니다. 실제 모델 실행은 아닙니다.
        structured_model = fixture_generation()
    else:
        # 라이브 모델 설정과 Schema를 준비하는 단계입니다. 아직 API를 호출하지 않습니다.
        chat_model = create_chat_model()
        # WireAnswer 스키마를 요청하고, 파싱 결과와 원본 메시지를 함께 받도록 설정합니다.
        structured_model = chat_model.with_structured_output(
            WireAnswer, method="json_schema", strict=True, include_raw=True)
    bundle = run_generation(prompt, structured_model, prepared["inputs"])
    if bundle is None:
        raise AssertionError("이번 정상 예시의 Context는 비어 있지 않아야 합니다.")
    parsed = bundle.get("parsed")
    cited_ids = list(parsed.source_ids) if parsed is not None else []
    # 응답의 인용 ID가 Context에 포함된 ID 집합 안에 있는지 기록합니다. 답변의 원문 대조는 별도 단계입니다.
    result = {
        "exercise": "06_context_chain",
        "execution": execution_record("offline_fixture" if args.offline else "live", llm=not args.offline),
        "real_model_calls": 0 if args.offline else 1,
        "context": prepared["context"],
        "included_ids": prepared["provided_ids"],
        "included_sources": [dict(doc.metadata) for doc in prepared["included_docs"]],
        "prompt_input": prepared["inputs"],
        "bundle": bundle_record(bundle),
        "observed_source_ids": cited_ids,
        "source_id_membership_only": None if parsed is None else set(cited_ids).issubset(prepared["provided_ids"]),
        "experiments": experiments,
        "human_evidence_review": "pending",
        "review_note": "bundle·포함 ID를 관찰한 기록입니다. 자료형·상태·출처의 승인 검사는 실습07에서 진행합니다.",
    }
    # 실행 모드, Context, 출처, 응답을 현재 실습 폴더의 outputs/context_chain.json에 저장합니다.
    path = save_json(HERE, "context_chain.json", result)
    show_success("06 Context·Runnable 관찰 기록", path)
    print("실제 LLM 실행:", not args.offline, "| 검증 승인 전 bundle 관찰")


if __name__ == "__main__":
    main()
