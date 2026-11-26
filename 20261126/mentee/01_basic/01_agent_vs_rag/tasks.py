"""01 · AI Agent vs 고정 RAG/Workflow · 기본필수.

실습 목적
    질문에 답하려면 문서 근거가 필요한지, DB에 저장된 현재 상태가 필요한지 구분합니다.
    예를 들어 인덱싱 기준은 문서에서 찾고, 63번 보고서의 현재 인덱싱 상태는 DB에서 읽습니다.

전체 흐름
    main.py가 예제 질문을 plan_for_request()에 전달합니다.
    plan_for_request()는 classify_request()로 전략을 정한 뒤 사용할 도구 이름을 담아 반환합니다.
    main.py는 계획을 출력하고, 별도 호출로 기존 report_agent_lab DB에서 상태를 조회합니다.
"""

def classify_request(question: str) -> str:
    """질문에 필요한 정보에 따라 'rag', 'db', 'agent' 중 하나를 반환합니다."""
    # TODO D29-01-B-01: 질문 성격을 'rag'·'db'·'agent' 중 선택. 규정+상태 복합은 agent.

    # READY처럼 대문자로 입력된 영문도 같은 키워드로 비교하도록 소문자로 바꿉니다.
    q=question.lower()
    # any()는 괄호 안 키워드 가운데 하나라도 질문에 포함되면 True를 반환합니다.
    # has_db는 상태 조회 단서, has_doc은 문서 검색 단서가 있는지를 나타냅니다.
    has_db=any(t in q for t in ('상태','인덱싱','revision','ready','pending'))
    has_doc=any(t in q for t in ('찾아','내용','설명','근거','검색'))

    # 키워드 기준이라 '인덱싱 기준을 설명해줘'처럼 문서 질문도 agent로 분류될 수 있습니다.
    # 두 단서가 모두 있는 질문도 아래의 DB 조건을 만족하므로, 복합 질문을 먼저 처리합니다.
    if has_db and has_doc:return 'agent'
    if has_db:return 'db'
    # DB 단서가 없으면 문서 검색을 기본 전략으로 선택합니다.
    return 'rag'

def plan_for_request(question: str) -> dict:
    """분류 결과를 바탕으로 처리 전략과 사용할 도구 이름을 반환합니다."""
    # TODO D29-01-B-02: 처리 전략과 사용할 Tool 목록을 반환합니다.

    # 분류값을 딕셔너리의 키로 사용해 도구 이름 목록을 고릅니다.
    # rag_search는 문서 검색, lookup_status는 보고서 상태 조회를 뜻합니다.
    mode=classify_request(question)
    tools={'rag':['rag_search'],'db':['lookup_status'],'agent':['rag_search','lookup_status']}[mode]
    # strategy는 분류값, tools는 도구 이름 목록, decision은 선택 방식에 대한 설명입니다.
    # 여기서는 고정 규칙으로 계획을 만들고, 06 실습에서 LLM이 도구를 선택합니다.
    return {'strategy':mode,'tools':tools,'decision':'고정 조건에 따른 실습용 계획; 실제 Agent 선택은 06에서 LLM 수행'}

