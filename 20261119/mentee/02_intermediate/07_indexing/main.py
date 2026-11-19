# 실습 목표 — 문서를 청크로 만들고, E5 임베딩을 pgvector에 저장해 질문으로 검색하는 과정을 익힙니다.
"""개별 실습 실행 진입점. TODO는 tasks.py에 있습니다."""
# 전체 흐름 — tasks.main()이 문서를 정제·분할하고 outputs/indexing_result.json에 결과를 저장합니다.
# --live-db를 주면 저장 전에 E5 임베딩 생성, DB 적재, 현재 버전 공개, 질문 검색까지 수행합니다.
from tasks import main

# python main.py로 직접 실행하면 아래의 main()이 호출됩니다.
if __name__ == '__main__':
    main()
