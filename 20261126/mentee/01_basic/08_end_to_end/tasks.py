"""08 · FastAPI → Agent → Tool → 응답 · 기본필수.

실습 목적
    Agent의 실행 결과를 검사하고 API에서 사용할 응답으로 정리합니다.

전체 흐름
    CLI는 기본 질문을 사용하고, FastAPI는 요청 본문에서 질문을 받습니다.
    두 경로 모두 run_agent()를 실행하고 api_response()로 완료 상태와 출처 ID를 확인합니다.
"""

def default_question() -> str:
    """실제 보고서 검색 + 63번 인덱싱 상태를 묻는 요청 문자열."""
    # TODO D29-08-B-01: 실제 보고서 검색 + 63번 인덱싱 상태를 묻는 요청 문자열.
    from coursekit.live import demo_question
    # CLI에서 사용할 공용 예제 질문을 가져옵니다.
    return demo_question()

def api_response(run: dict) -> dict:
    """Agent가 정상 완료했고 출처 검증을 통과했으면 응답 딕셔너리를 반환합니다."""
    # TODO D29-08-B-02: 성공/실패와 최종 답변·출처 검증을 API 계약으로 변환.
    # 먼저 Agent가 최종 답변을 만들었는지 완료 상태로 확인합니다.
    if run.get('status')!='completed':raise RuntimeError('Agent did not complete')
    # 답변에 인용된 RPT 번호가 모두 실제 도구 결과에 들어 있었는지 run_agent()가 검사합니다.
    # 여기서는 그 결과가 True일 때만 정상 응답을 만듭니다.
    if run.get('citation_ids_valid') is not True:
        raise RuntimeError('UNVERIFIED_CITATION: Agent가 제공되지 않은 출처 ID를 인용했습니다.')
    # 응답에는 최종 답변(answer), 도구 실행 기록(tool_trace), 확인된 근거 ID를 담습니다.
    return {'status':'completed','answer':run['answer'],
            'tool_trace':run['tool_trace'],
            'citation_ids_valid':run['citation_ids_valid'],
            'verified_evidence_ids':run['verified_evidence_ids']}

