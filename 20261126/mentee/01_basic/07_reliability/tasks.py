"""07 · Tool 실패·권한·반복 상한 · 기본필수.

실습 목적
    도구에 전달할 보고서 ID를 검증하고, 예외를 호출자가 처리할 수 있는 오류 코드로 바꿉니다.

전체 흐름
    main.py에서 기존 report_agent_lab의 63번 보고서를 읽어 정상 조회를 확인합니다.
    이어서 예제 ID를 검사하고, 잘못된 입력에서 발생한 예외를 오류 코드로 출력합니다.
    Agent 실행 중 호출 횟수를 제한하는 코드는 06_create_agent의 미들웨어에서 확인할 수 있습니다.
"""

def validate_report_id(value) -> int:
    """DB 조회에 사용할 보고서 ID를 양의 정수로 검사합니다."""
    # TODO D29-07-B-01: 조회 전에 보고서 ID 형식을 검증합니다.
    from coursekit.contracts import report_id_from
    # 공용 함수는 '63'을 정수 63으로 변환하고, -1처럼 허용 범위를 벗어나면 ValueError를 발생시킵니다.
    return report_id_from(value)

def classify_error(exc: Exception) -> dict:
    """전달받은 예외의 종류에 따라 오류 응답을 만듭니다."""
    # TODO D29-07-B-02: 입력 오류와 그 밖의 도구 실패를 오류 코드로 구분합니다.
    # ValueError는 INVALID_ARGS로, 함수에 전달된 다른 예외는 TOOL_FAILURE로 분류합니다.
    # 예외 메시지에 접속 정보나 SQL이 들어 있을 수 있어, 반환값에는 상태와 오류 코드만 담습니다.
    return {'status':'error','error_code':'INVALID_ARGS' if isinstance(exc,ValueError) else 'TOOL_FAILURE'}

