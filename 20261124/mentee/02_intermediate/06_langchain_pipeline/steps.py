"""단계별 역할 분리: steps.py / 독립 실행은 main.py. """

# 실습 질문은 '그럼 취소는 어떻게 해?'입니다. 앞 대화의 주제는 '회의실 예약'입니다.
# 아래 첫 함수는 질문과 앞 대화를 한 묶음으로 돌려줍니다. 이 묶음이 실행할 요청입니다.
# 둘째 함수는 그 요청을 받아 검색하고 답변할 수 있도록 작업 순서를 연결해서 돌려줍니다.

def prepare_pipeline_request(case):
    """원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다."""
    # BEGIN D27-06-I01
    # TODO D27-06-I01: 원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # 1. case는 실습 문제 한 건입니다. case['question']에서 '그럼 취소는 어떻게 해?'를 꺼내
    #    새 딕셔너리의 query 칸에 넣습니다. history에는 앞 대화, category에는 '시설'을 넣습니다.
    # 2. 숫자 5와 2는 각각 '최대 5개 검색'과 '그중 최대 2개를 답변 근거로 사용'이라는 뜻입니다.
    # 3. True는 기능을 켠다는 뜻입니다. 이 요청을 실행하면 질문을 검색에 맞게 다시 쓰고
    #    검색 결과를 관련도 순으로 정렬합니다. use_multiquery=False라서 검색어는 하나만 씁니다.
    # 이 return이 돌려주는 값은 질문·대화·설정을 묶은 요청입니다. query에는 원래 문장이 남습니다.
    return {'query':case['question'],'history':case['history'],
            'category':case['category'],'candidate_k':5,'context_k':2,
            'use_rewrite':True,'use_multiquery':False,'use_rerank':True}
    # END D27-06-I01


def wire_langchain_pipeline(dependencies, offline):
    """LCEL는 Runnable을 준비할 뿐이고 .invoke()가 실행 시작점입니다."""
    # BEGIN D27-06-I02
    # TODO D27-06-I02: LCEL는 Runnable을 준비할 뿐이고 .invoke()가 실행 시작점입니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # dependencies는 (질문 변환 함수, 검색 함수, 재정렬 함수, 답변 함수)처럼 함수 네 개를 묶은 값입니다.
    # *dependencies는 묶음에서 네 함수를 하나씩 꺼내 create_rag_pipeline()에 순서대로 넘기는 문법입니다.
    # 이 return이 돌려주는 값은 네 함수를 연결한 체인입니다. 실제 작업은 실행 코드가
    # chain.invoke(req)로 첫 함수에서 만든 요청을 넣는 순간 시작합니다.
    #
    # 요청을 넣었을 때 실제 오프라인 실행에서 일어난 일
    # 1. 앞 대화에 '회의실'이 있는지 확인하고 '회의실 예약을 취소하는 방법은?'으로 검색어를 만듭니다.
    # 2. 시설 문서에서 후보 5개를 가져와 교육용 고정 점수 순으로 정렬합니다.
    # 3. 그중 FAC-102-C1, FAC-103-C1의 본문을 골라 답변에 사용할 근거로 넘깁니다.
    # 4. 답변, 인용한 출처, 실행 기록을 묶어 돌려줍니다. checks.py는 이 기록을 받아 검사합니다.
    from coursekit.pipeline import create_rag_pipeline
    return create_rag_pipeline(*dependencies,offline=offline)
    # END D27-06-I02
