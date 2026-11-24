"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다."""

# 실습 목적은 검색 결과(Hit)를 LangChain Document로 변환하면서 원문과 출처 정보를 보존하는 것입니다.
# 예시나 실제 검색에서 Hit를 가져온 뒤 Document로 변환하고, 원본과 대조한 결과를 JSON에 저장합니다.

from pathlib import Path
import sys

# 현재 파일 경로에서 실습 폴더와 멘티 루트를 찾아 공통 지원 모듈을 불러옵니다.
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import save_json, show_success
from coursekit.data import example_hit, load_hits
from coursekit.reporting import execution_record
from langchain_core.documents import Document
import argparse

def metadata_of(hit: dict) -> dict:
    """본문과 분리해서 여섯 Metadata 필드를 그대로 반환합니다."""
    # BEGIN D26-04-I01
    # TODO D26-04-I01: 보존할 여섯 Key를 정하고 입력의 값을 그대로 반환합니다.
    # 힌트: citation_id·chunk_id·doc_id·source_version·source_locator·category입니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 문서와 청크의 식별자, 원문 버전과 위치 등 출처를 추적할 여섯 필드를 고릅니다.
    keys = ["citation_id", "chunk_id", "doc_id", "source_version", "source_locator", "category"]

    # 입력값을 그대로 복사하므로 정수로 들어온 citation_id의 자료형도 유지됩니다.
    # source_locator 같은 필수 키가 빠지면 KeyError가 발생합니다.
    return {key: hit[key] for key in keys}
    # END D26-04-I01

def to_document(hit: dict) -> Document:
    """원문 본문을 그대로 두고 출처 Metadata와 함께 Document를 생성합니다."""
    # BEGIN D26-04-I02
    # TODO D26-04-I02: 본문과 metadata_of의 결과로 Document를 생성합니다.
    # 힌트: hit의 body는 수정하지 않습니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 원문 전체는 page_content에 넣고, 출처 정보는 metadata에 담아 Document 한 건을 만듭니다.
    return Document(page_content=hit["body"], metadata=metadata_of(hit=hit))
    # END D26-04-I02

def hits_to_documents(hits: list[dict]) -> list[Document]:
    """같은 순서의 Document 목록을 반환합니다."""
    # BEGIN D26-04-I03
    # TODO D26-04-I03: 입력 Hit들을 순서대로 변환합니다.
    # 힌트: to_document를 호출해 빈 목록부터 차례로 채웁니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # Hit마다 Document를 하나씩 추가해 입력 순서를 유지합니다. 입력이 비어 있으면 빈 목록이 반환됩니다.
    results = []

    for hit in hits:
        results.append(to_document(hit=hit))
    
    return results
    # END D26-04-I03

def metadata_is_preserved(hit: dict, metadata: dict) -> bool:
    """제공된 관찰 함수: 여섯 필드가 모두 있고 각 값이 같은지 확인합니다."""
    required = {"citation_id", "chunk_id", "doc_id", "source_version", "source_locator", "category"}
    # 일부 필드만 복사한 결과를 '보존 완료'로 표시하지 않도록 Key 집합부터 확인합니다.
    # 여섯 키가 정확히 일치하면 all()로 각 값이 원본 Hit의 값과 같은지 확인합니다.
    return set(metadata) == required and all(metadata[key] == hit[key] for key in required)


def main():
    # 명령줄 옵션으로 입력 경로를 선택합니다. 질문·분류·후보 수는 실제 검색에 전달됩니다.
    parser = argparse.ArgumentParser(description="Hit → 실제 LangChain Document 변환")
    parser.add_argument("--live-db", action="store_true", help="실제 E5·PostgreSQL 검색 결과를 입력으로 사용")
    parser.add_argument("--question", default="15분 늦으면 어떻게 기록되나요?")
    parser.add_argument("--category", default="학사", choices=["학사", "시설", "지원"])
    parser.add_argument("--candidate-k", type=int, default=3)
    args = parser.parse_args()
    if args.live_db:
        # --live-db를 선택한 경우에만 실제 모델·DB 준비 함수를 불러옵니다.
        from coursekit.live import make_search_hits
        search_hits = make_search_hits()
        hits = search_hits(args.question, args.category, k=args.candidate_k)
        mode, input_source = "live_db", "E5_PostgreSQL"
    else:
        # 기본 실행은 공통 예시 데이터의 첫 번째 Hit 한 건으로 Document 변환을 확인합니다.
        hits = [example_hit()]
        mode, input_source = "local_adapter", "lecture_fixture"
    documents = hits_to_documents(hits)
    comparisons = []
    # 변환 순서를 유지하므로 zip()으로 각 Hit와 해당 Document를 짝지어 본문과 출처를 대조합니다.
    for hit, doc in zip(hits, documents):
        comparisons.append({
            "citation_id": doc.metadata["citation_id"],
            "body_preserved": doc.page_content == hit["body"],
            "metadata_preserved": metadata_is_preserved(hit, doc.metadata),
            # 인용 ID가 정수로 보존됐는지 결과 파일에서 확인할 수 있도록 자료형 이름을 기록합니다.
            "citation_id_type": type(doc.metadata["citation_id"]).__name__,
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        })
    report = {
        # 실행 모드와 입력 출처를 남겨 예시 사용 여부와 실제 검색 실행 여부를 확인할 수 있습니다.
        "execution": execution_record(mode, db=args.live_db, embedding=args.live_db),
        "input_source": input_source,
        "input_count": len(hits), "document_count": len(documents),
        "documents": comparisons,
    }
    # 현재 실습 폴더의 outputs/document_adapter.json에 저장하고 터미널에 저장 경로를 표시합니다.
    path = save_json(HERE, "document_adapter.json", report)
    show_success("04 응용", path)

# tasks.py를 직접 실행하면 main.py와 같은 main()을 호출합니다.
if __name__ == "__main__":
    main()
