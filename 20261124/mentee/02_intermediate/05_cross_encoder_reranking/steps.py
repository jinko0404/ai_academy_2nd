"""단계별 역할 분리: steps.py / 독립 실행은 main.py. """

# 실습 목적
# 각 후보의 본문이 질문과 얼마나 관련 있는지 점수를 매겨 검색 순서를 다시 정합니다.

# 전체 흐름
# create_pairs()는 모델 입력이 어떻게 생겼는지 보여 주는 예시를 만드는 데 쓰입니다.
# rank_documents()는 같은 질문과 후보 문서를 받아 점수가 가장 높은 문서를 돌려줍니다.
from coursekit.reranking import rerank_documents

def create_pairs(question, docs):
    """CrossEncoder 입력은 [질문, 문서본문]의 쌍 목록입니다."""
    # BEGIN D27-05-I01
    # TODO D27-05-I01: CrossEncoder 입력은 [질문, 문서본문]의 쌍 목록입니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # docs는 검색된 Document(문서) 목록이고, page_content는 각 문서의 본문입니다.
    # 문서 하나마다 [질문, 본문] 리스트 하나를 만듭니다. 질문은 그대로 두고 본문만 바뀝니다.
    # 후보가 두 개면 [[질문, 첫 번째 본문], [질문, 두 번째 본문]] 모양으로 반환합니다.
    pairs = []
    for doc in docs:
        pairs.append([question, doc.page_content])
    return pairs
    # END D27-05-I01


def rank_documents(question, docs, scorer):
    """Reranker는 후보를 새로 찾지 않고 동일 후보를 순위 변경합니다."""
    # BEGIN D27-05-I02
    # TODO D27-05-I02: Reranker는 후보를 새로 찾지 않고 동일 후보를 순위 변경합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # 공통 함수가 질문과 본문을 짝지어 scorer에 넘깁니다. scorer는 입력 순서대로
    # 문서마다 점수 하나를 돌려줍니다. 공통 함수는 점수 개수가 후보 수와 같은지 확인합니다.
    # 문서를 복사해 출처 정보는 보존하고 metadata['rerank_score']에 점수를 기록합니다.
    # 점수가 큰 순서로 정렬해 최대 3개를 반환합니다. 동점이면 검색 순서를 유지합니다.
    return rerank_documents(question, docs, scorer, top_n=3)
    # END D27-05-I02
