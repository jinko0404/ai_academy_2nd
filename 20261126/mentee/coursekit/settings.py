"""11/20 기존 report_agent_lab 연결. 새 DB 생성·테이블 변경을 하지 않습니다."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('EXPERIMENT', 'DAILY_WORK', 'RESEARCH_PLAN', 'DATA_PROCESSING', 'PROMPT', 'MEETING')

@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model: str
    reranker_model: str
    llm_model: str
    cuda_device: str
    max_tool_calls: int
    min_similarity: float
    allowed_categories: tuple[str, ...]

@lru_cache(maxsize=1)
def settings() -> Settings:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env')
    except ImportError:
        pass
    url = os.getenv('DATABASE_URL', '').strip()
    if not url:
        raise RuntimeError('DATABASE_URL을 .env에 지정하세요. 기존 report_agent_lab 접속 문자열이 필요합니다.')
    if not os.getenv('LLM_MODEL', '').strip():
        model = ''  # DB-only 과제는 LLM 설정이 없어도 가능. Agent 실행시에만 검사
    else:
        model = os.environ['LLM_MODEL'].strip()
    allowed = tuple(x.strip().upper() for x in os.getenv('ALLOWED_CATEGORIES', 'EXPERIMENT,DAILY_WORK,RESEARCH_PLAN,DATA_PROCESSING,PROMPT,MEETING').split(',') if x.strip())
    if not allowed or not set(allowed).issubset(CATEGORIES):
        raise ValueError('ALLOWED_CATEGORIES에 존재하지 않는 Category가 있습니다.')
    device = os.getenv('CUDA_DEVICE', 'cuda:0')
    if not device.startswith('cuda:'):
        raise ValueError('Day29는 CUDA GPU 필수입니다. CUDA_DEVICE=cuda:0을 사용하세요.')
    return Settings(
        database_url=url,
        embedding_model=os.getenv('EMBEDDING_MODEL','intfloat/multilingual-e5-small').strip(),
        reranker_model=os.getenv('RERANKER_MODEL','BAAI/bge-reranker-v2-m3').strip(),
        llm_model=model,
        cuda_device=device,
        max_tool_calls=int(os.getenv('MAX_TOOL_CALLS','4')),
        min_similarity=float(os.getenv('MIN_COSINE_SIMILARITY','0.55')),
        allowed_categories=allowed,
    )

def require_llm_settings():
    s=settings()
    if not s.llm_model:
        raise RuntimeError('LLM_MODEL이 비어 있습니다. Tool Calling 지원 모델을 .env에 지정하세요.')
    if not os.getenv('OPENAI_API_KEY','').strip():
        raise RuntimeError('OPENAI_API_KEY가 비어 있습니다. 실제 API 호출 시 인증이 필요합니다.')
    return s
