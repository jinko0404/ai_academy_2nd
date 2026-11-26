"""각 실습의 공통 실행 부품. --live는 실제 PostgreSQL·GPU·LLM을 호출합니다."""
from coursekit.db import preflight,fetch_report_status,fetch_report_read_view
from coursekit.gpu import require_cuda
from coursekit.rag import retrieve_reports,rag_answer
from coursekit.agent import run_agent,build_tools,build_agent


def demo_question()->str:
    return 'RAG 근거 인용 정확도 향상과 관련된 보고서의 내용을 찾아 출처를 제시하고, 보고서 63번의 인덱싱 상태도 알려줘.'
