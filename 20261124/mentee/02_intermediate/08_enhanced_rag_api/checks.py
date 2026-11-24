"""단계별 역할 분리: checks.py / 독립 실행은 main.py. """

# API가 보내온 응답을 열어 성공이면 답변에 달린 출처 번호를 확인하고, 오류이면 오류 내용을 꺼냅니다.
# 검사 결과는 저장된 JSON의 각 rows 항목 안에 response_check로 기록됩니다.

def check_http_response(response):
    """성공 응답의 source_ids와 실제 연결된 sources의 ID가 같은지 검사합니다."""
    # BEGIN D27-08-I03
    # TODO D27-08-I03: 성공 응답의 source_ids와 실제 연결된 sources의 ID가 같은지 검사합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # 1. response.json()은 응답 본문을 파이썬 딕셔너리로 읽습니다.
    body=response.json()

    # 2. HTTP 200 이외의 응답은 오류 분기로 갑니다. 예를 들어 422는 요청 값이 잘못된 경우,
    #    502는 이 실습에서 체인 단계의 실패를 확인하는 경우입니다. detail에 담긴 내용을 반환합니다.
    if response.status_code!=200:return {'http_ok':False,'error_detail':body.get('detail')}

    # 3. HTTP 200이면 실제 출처 목록(sources)에서 citation_id를 하나씩 꺼냅니다.
    #    이 번호들이 답변에 표시한 번호(source_ids)와 값과 순서가 모두 같으면 True입니다.
    #    오프라인 Q01에서는 양쪽이 [1002]입니다. Q06은 근거가 부족해 양쪽이 []이고, 이때도 비교는 True입니다.
    ids=[src['citation_id'] for src in body['sources']]
    return {'http_ok':True,'citation_consistent':ids==body['result']['source_ids']}
    # END D27-08-I03
