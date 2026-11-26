"""03 · @tool·Pydantic·JSON Schema · 기본필수.

실습 목적
    Python의 보고서 상태 조회 함수를 LangChain 도구로 등록합니다.
    함수 이름과 설명, 인자 이름과 자료형이 도구를 호출할 때 필요한 정보를 제공합니다.

전체 흐름
    main.py에서 예제 ID의 형식을 확인한 뒤 도구를 만들고 입력 스키마를 출력합니다.
    도구의 invoke()가 호출되면 기존 report_agent_lab DB에서 상태를 읽어 JSON 문자열로 반환합니다.
"""

def validate_id(report_id: int) -> int:
    """보고서 ID의 형식을 검사하고 양의 정수로 반환합니다."""
    # TODO D29-03-B-01: DB 조회에 사용할 report_id를 양의 정수로 검증합니다.
    # 공용 함수는 63과 '63'을 정수 63으로 처리합니다. 보고서 존재 여부는 DB 조회에서 확인합니다.
    from coursekit.contracts import report_id_from
    return report_id_from(report_id)

def register_status_tool():
    """보고서 상태 조회 함수를 LangChain 도구로 만들어 반환합니다."""
    # TODO D29-03-B-02: @tool을 사용해 실제 DB 상태 조회 함수를 등록하고 반환합니다.

    # 실습 가상환경에 설치된 langchain-core에서 @tool 데코레이터를 가져옵니다.
    from langchain_core.tools import tool
    import json
    from coursekit.db import fetch_report_status

    # @tool은 함수 이름 lookup_status와 docstring을 도구의 이름과 설명으로 사용합니다.
    # report_id: int는 Pydantic 입력 스키마(args_schema)에서 필수 정수 인자가 됩니다.
    @tool
    def lookup_status(report_id:int)->str:
        """기존 보고서 report_id의 인덱싱 상태를 읽기 전용 조회합니다. task_id가 아닙니다."""
        # 실행 시 ID를 검사하고 SELECT로 상태를 읽은 다음, 조회 결과를 JSON 문자열로 바꿉니다.
        # ensure_ascii=False는 한글을 그대로 표시하고, default=str은 JSON 변환을 지원하지 않는 값을 문자열로 처리합니다.
        return json.dumps(fetch_report_status(validate_id(report_id)),ensure_ascii=False,default=str)
    # 호출하는 쪽에 도구 객체를 돌려주면 스키마 확인과 invoke() 실행에 사용할 수 있습니다.
    return lookup_status

