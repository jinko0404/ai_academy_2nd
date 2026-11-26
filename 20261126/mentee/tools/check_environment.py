"""DB, GPU를 실제로 확인하는 진단. 네트워크 API 호출이나 DB 변경은 하지 않습니다."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coursekit.settings import settings
from coursekit.db import preflight
from coursekit.gpu import require_cuda


def main():
    s=settings()
    print('기존 DB:',s.database_url.rsplit('@',1)[-1])  # 비밀번호 출력 금지
    print('CUDA:',require_cuda())
    result=preflight()
    print('PostgreSQL 사전 점검:',result)
    if not result['vector_active'] or result['vectors']<1 or result['matching_model_vectors']<1 or result['fresh_vectors']<1:
        print('주의: 기존 11/20 DB에서 pgvector/Embedding 완료 여부를 확인하세요. DB를 새로 만들지 않습니다.')
    else:print('기존 DB 및 E5 인덱싱을 읽기 전용으로 확인했습니다.')
    if not s.llm_model:print('LLM_MODEL 미설정: 실습 06~08의 실제 Agent 실행 전 설정하세요.')

if __name__=='__main__':main()
