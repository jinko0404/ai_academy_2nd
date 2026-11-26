"""02 · AIMessage.tool_calls / ToolMessage · 기본필수.

실습 목적
    도구 호출 요청에는 무엇을 실행할지, 결과 메시지에는 실행해서 얻은 값이 담깁니다.
    요청의 id를 결과의 tool_call_id에 넣으면 여러 호출 중 어느 요청에 대한 결과인지 알 수 있습니다.

전체 흐름
    main.py는 63번 보고서를 실제 DB에서 조회한 뒤 예제 호출 요청을 만듭니다.
    match_result()는 이 요청과 조회 결과를 받아 ToolMessage 형식의 실습용 딕셔너리를 만듭니다.
"""

def make_request(tool_name: str, args: dict, call_id: str) -> dict:
    """AIMessage.tool_calls 목록의 항목 하나에 해당하는 요청을 만듭니다."""
    # TODO D29-02-B-01: 도구 이름, 인자, 호출 ID를 요청 딕셔너리로 만듭니다.
    # name은 도구 이름, args는 도구에 넘길 값, id는 이번 호출을 식별하는 값입니다.
    # 예제에서는 args에 report_id=63을, id에 call_001을 넣습니다.
    from coursekit.contracts import tool_call_request
    # 공용 함수가 등록된 도구 이름, args의 자료형, 비어 있지 않은 호출 ID를 검사합니다.
    return tool_call_request(tool_name,args,call_id)

def match_result(call: dict, result: dict) -> dict:
    """이미 얻은 조회 결과를 해당 요청에 연결하는 딕셔너리를 만듭니다."""
    # TODO D29-02-B-02: 요청과 조회 결과를 받아 ToolMessage 형식으로 묶습니다.
    from coursekit.contracts import tool_result_message
    # 공용 함수가 call['id']를 tool_call_id로 복사하고 result를 content에 담습니다.
    # 결과 딕셔너리에는 메시지 유형(type)과 요청에 있던 도구 이름(name)도 들어갑니다.
    return tool_result_message(call,result)

