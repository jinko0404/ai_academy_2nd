"""단계별 역할 분리: steps.py / 독립 실행은 main.py. """

# 실습 목적
# 실습 문제의 질문과 분류를 검색기가 받을 수 있는 요청으로 바꾸고,
# 검색 결과로 받은 후보 문서의 개수가 요청한 한도 안에 있는지 확인합니다.

# 전체 흐름
# 실습 문제(case) → 검색 요청(payload) → 검색기(Retriever) → 후보 문서 목록(docs)
# 후보 목록과 검색 요청은 이어서 checks.py의 metadata_check()에 전달됩니다.

def make_candidate_request(case):
    """candidate_k와 context_k는 다른 제한값입니다."""
    # BEGIN D27-04-I01
    # TODO D27-04-I01: candidate_k와 context_k는 다른 제한값입니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # case는 실습 문제의 정보가 담긴 딕셔너리입니다. question 값을 검색어(query)로,
    # category 값을 검색할 문서 분류로 사용해 새 요청 딕셔너리를 만들어 반환합니다.
    # candidate_k=5는 검색에서 가져올 후보의 최대 개수입니다.
    # context_k=2는 답변에 제공할 참고 문맥을 만들 때 쓸 문서의 최대 개수입니다.
    # 04번 실습에서는 이 값을 결과에 기록하고, 실제 참고 문맥 구성은 이후 단계에서 다룹니다.
    return {'query':case['question'],'category':case['category'],
                'candidate_k':5,'context_k':2}
    # END D27-04-I01


def fetch_documents(retriever, payload):
    """Day26에서 배운 Runnable.invoke로 후보 Document를 받아옵니다."""
    # BEGIN D27-04-I02
    # TODO D27-04-I02: Day26에서 배운 Runnable.invoke로 후보 Document를 받아옵니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # invoke(payload)는 요청 딕셔너리를 넘겨 검색을 한 번 실행하는 메서드입니다.
    # 검색 결과인 Document 목록에는 각 문서의 본문과 출처 등의 메타데이터가 들어 있습니다.
    docs=retriever.invoke(payload)

    # len(docs)는 받은 후보 수입니다. 이 수가 요청한 최대 개수보다 크면 예외로 알려 줍니다.
    # 한도 이내라면 받은 문서 목록을 그대로 돌려줍니다.
    if len(docs)>payload['candidate_k']:raise ValueError('candidate_k 초과')
    return docs
    # END D27-04-I02
