"""단계별 역할 분리: checks.py / 독립 실행은 main.py. """

# 실습 목적
# 검색 결과를 답변의 근거로 쓸 수 있도록 각 후보 문서에 출처 정보가 남아 있는지,
# 그리고 요청한 분류(category)에 해당하는지 확인합니다.
# Document(후보 문서)는 본문(page_content)과 부가 정보(metadata)로 구성됩니다.
# metadata는 딕셔너리이며, chunk_id는 문서 조각의 ID, source_locator는 원문 위치입니다.
# 예를 들어 원문 위치에는 페이지나 조항을 나타내는 정보가 들어갈 수 있습니다.

# 전체 흐름
# 1. main.py가 실습을 시작하면 steps.py의 make_candidate_request()가
#    질문, 분류, 검색 후보 수 제한 등을 담은 요청 딕셔너리(payload)를 만듭니다.
# 2. steps.py의 fetch_documents()가 검색기(Retriever)에 요청을 보내
#    후보 문서 목록(docs)을 받고, 후보 수가 요청한 한도를 넘었는지 검사합니다.
# 3. 아래 metadata_check()가 docs와 payload를 받아 문서의 형식과 분류를 확인합니다.
#    실습 실행 코드는 반환값을 결과 JSON의 row.metadata_ok에 기록합니다.

def metadata_check(docs, payload):
    """각 Document의 chunk_id·source_locator·Category를 검사합니다."""
    # BEGIN D27-04-I03
    # TODO D27-04-I03: 각 Document의 chunk_id·source_locator·Category를 검사합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # 공통 검사 함수는 본문이 비었거나 필수 메타데이터 항목이 빠졌을 때,
    # 또는 인용 ID(citation_id)가 양의 정수가 아닐 때 ValueError를 발생시킵니다.
    # 검사를 통과하면 받은 문서를 그대로 돌려줍니다.
    from coursekit.documents import valid_document

    # for d in docs는 후보를 하나씩 d에 담습니다. 각 후보는 먼저 공통 검사를 거친 뒤
    # 문서의 분류와 요청의 분류(payload['category'])를 ==로 비교합니다.
    # all()은 모든 비교가 참이면 True를 반환합니다. 분류가 다른 후보를 만나면
    # False를 즉시 반환하므로 이후 후보는 검사하지 않습니다.
    # 검사 도중 발생한 ValueError는 호출한 쪽으로 전달됩니다.
    # 후보 목록이 비어 있으면 비교할 항목이 없어 all()은 True를 반환합니다.
    return all(valid_document(d).metadata['category']==payload['category']
               for d in docs)
    # END D27-04-I03
