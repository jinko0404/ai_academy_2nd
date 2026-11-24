"""단계별 역할 분리: steps.py / 독립 실행은 main.py. """

# 첫 함수는 질문에 답하는 작업 순서(chain)를 만들어 반환합니다.
# 둘째 함수는 그 chain을 API 요청이 들어올 때 실행하도록 연결한 FastAPI 앱을 반환합니다.

def make_service_chain(offline):
    """Rewrite → Retrieve → Rerank → Context → Generation·Validation을 연결합니다."""
    # BEGIN D27-08-I01
    # TODO D27-08-I01: Rewrite → Retrieve → Rerank → Context → Generation·Validation을 연결합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # 1. create_demo_chain()은 질문 다듬기 → 문서 검색 → 관련도 순 정렬 → 답변 근거 고르기
    #    → 답변 생성과 출처 확인을 잇습니다. offline 값이 교육용 또는 실제 실행 방식을 결정합니다.
    #    여기서 만든 chain은 아래 wire_fastapi()의 입력이 됩니다.
    from coursekit.runtime import create_demo_chain
    return create_demo_chain(offline=offline)
    # END D27-08-I01


def wire_fastapi(chain):
    """API 요청의 role 문자열을 신뢰하지 않고 서버에서 learner 역할을 지정합니다."""
    # BEGIN D27-08-I02
    # TODO D27-08-I02: API 요청의 role 문자열을 신뢰하지 않고 서버에서 learner 역할을 지정합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # 2. create_app()은 POST /api/v1/rag/answer를 등록합니다. 이 실습에서는 서버 코드가
    #    검색 권한을 learner(수강생)로 고정해 공개 문서 범위에서 검색합니다.
    # 3. 유효한 요청이 들어오면 coursekit/api.py의 answer()가 요청을 딕셔너리(body)로 바꾸고
    #    chain.invoke(body)로 실행합니다. 체인이 돌려준 답변과 출처가 API의 JSON 응답이 됩니다.
    #    예를 들어 candidate_k=0은 입력 검증에서 걸려 체인 실행 전에 HTTP 422가 반환됩니다.
    from coursekit.api import create_app
    return create_app(chain,trusted_role='learner')
    # END D27-08-I02
