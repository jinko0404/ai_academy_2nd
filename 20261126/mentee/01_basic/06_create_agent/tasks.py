"""06 · create_agent + 실제 Tool 2개 · 기본필수.

실습 목적
    실제 LLM이 질문을 읽고 등록된 도구 중 필요한 것을 선택하는 과정을 확인합니다.
    rag_search는 보고서 내용을 검색하고, lookup_status는 특정 보고서의 인덱싱 상태를 읽습니다.

전체 흐름
    예제 질문을 만들고, 모델과 두 도구를 등록한 Agent를 main.py에서 실행합니다.
    모델이 AIMessage.tool_calls로 호출을 요청하면 실행 결과가 ToolMessage로 돌아옵니다.
    모델은 결과를 참고해 다음 도구를 요청하거나 최종 답변을 작성합니다.
"""

def question_for_agent(report_id: int) -> str:
    """내용 검색 + 인덱싱 상태를 모두 묻는 자연어 질문을 만듭니다."""
    # TODO D29-06-B-01: 내용 검색 + 인덱싱 상태를 모두 묻는 자연어 질문을 만듭니다.
    from coursekit.contracts import report_id_from
    # 보고서 ID를 양의 정수로 검증하고 f-string으로 질문에 삽입합니다.
    # 질문에 두 요청을 담아도 실제 도구 선택과 호출 순서는 모델이 결정합니다.
    rid=report_id_from(report_id)
    return f"RAG 근거 인용 정확도 향상과 관련된 보고서 내용을 설명하고, {rid}번 보고서의 인덱싱 상태도 알려줘."

def make_agent():
    """모델, 도구, 호출 제한을 설정한 Agent를 반환합니다."""
    # TODO D29-06-B-02: create_agent + 등록 Tool 2개 + LLM Model + Tool Call 상한을 설정.
    # build_tools()는 두 도구를 만들고, SYSTEM_PROMPT는 모델에 사용 가능한 도구와 응답 규칙을 안내합니다.
    from coursekit.agent import build_tools,SYSTEM_PROMPT
    from coursekit.settings import require_llm_settings,settings
    # create_agent와 미들웨어는 langchain 패키지에 들어 있습니다. 실습 요구 버전은 ../../requirements.txt에 있습니다.
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware,ModelCallLimitMiddleware
    from langchain_openai import ChatOpenAI
    # .env 등에서 읽은 LLM_MODEL과 OPENAI_API_KEY가 설정되어 있는지 확인합니다.
    require_llm_settings()
    # 한 번의 Agent 실행에서 도구 호출은 최대 4회, 모델 호출은 최대 6회로 제한합니다.
    # exit_behavior='error'는 상한을 넘기는 호출을 오류로 처리합니다.
    return create_agent(model=ChatOpenAI(model=settings().llm_model,timeout=45,max_retries=1),
        tools=build_tools(),system_prompt=SYSTEM_PROMPT,
        middleware=[ToolCallLimitMiddleware(run_limit=4,exit_behavior='error'),
                    ModelCallLimitMiddleware(run_limit=6,exit_behavior='error')])

