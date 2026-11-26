"""04 · 기존 RAG를 rag_search Tool로 · 기본필수.

실습 목적
    기존 report_agent_lab에 저장된 보고서를 검색해 근거가 있는 답변을 만듭니다.
    이 실습의 main.py는 검색 함수를 직접 호출하고, Agent의 rag_search 도구도 같은 RAG 함수를 사용합니다.

전체 흐름
    검색어와 분류를 정리한 뒤 CUDA의 E5 모델로 검색어를 벡터로 변환합니다.
    pgvector가 기존 보고서 요약 벡터에서 후보를 찾고, CUDA 재정렬 모델이 관련도 순서를 다시 매깁니다.
    답변에 쓸 보고서 본문이 확보되면 LLM에 근거로 전달합니다.
"""

def prepare_search(query: str, category: str = 'ALL') -> dict:
    """검색어와 카테고리를 검증하고 검색에 사용할 딕셔너리를 반환합니다."""
    # TODO D29-04-B-01: 검색 입력을 검증하고 query와 category를 반환합니다.
    from coursekit.contracts import search_request
    # 공용 함수가 검색어 양쪽 공백을 제거하고 길이를 1~500자로 제한합니다.
    # 카테고리는 공백을 제거하고 대문자로 바꿉니다. 등록된 6개 분류와 전체 검색을 뜻하는 ALL을 허용합니다.
    query,category=search_request(query,category)
    return {'query':query,'category':category}

def invoke_rag(request: dict) -> dict:
    """검증된 검색 입력으로 기존 DB와 CUDA를 사용하는 RAG를 실행합니다."""
    # TODO D29-04-B-02: 실제 RAG 함수에 검색어와 카테고리를 전달합니다.
    from coursekit.rag import rag_answer
    # 결과에는 처리 상태(status), 답변(answer), 출처(sources), 검색 과정의 보고서 ID(retrieval)가 담깁니다.
    # 근거를 확보하지 못하면 LLM 호출 전에 insufficient_context 상태와 빈 출처 목록을 반환합니다.
    return rag_answer(request['query'],request['category'])

