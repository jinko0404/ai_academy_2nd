# Day23 / 실습 06 / 응용 / mentee
# 인덱스 생성과 실행 계획 조회에 쓸 SQL을 준비합니다.
# 이어서 계획의 스캔 방식, 코사인 거리 설정, 결과 ID의 일치율을 확인합니다.
from day23kit.db import HNSW_SQL

# BEGIN D23-06-I01
def create_index_sql():
    """
    TODO D23-06-I01 · create_index_sql
        day23_search_docs.embedding의 Cosine HNSW 인덱스를 정의하세요.
        USING hnsw (embedding vector_cosine_ops)를 사용합니다.
        인덱스는 선택 항목이며 9건에서 빠른 성능을 보장하지 않습니다.
    """
    # TODO D23-06-I01: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # SQL 문자열을 돌려줍니다. vector_cosine_ops는 인덱스에 코사인 거리 기준을 지정합니다.
    return HNSW_SQL
# END D23-06-I01

# BEGIN D23-06-I02
def explain_sql():
    """
    TODO D23-06-I02 · explain_sql
        SELECT id FROM day23_search_docs를 거리순으로 검색하는
        EXPLAIN (ANALYZE, BUFFERS) SQL을 작성하세요. 바인딩은 %s 두 개.
        인덱스 사용 여부를 임의로 주장하면 안 됩니다.
    """
    # TODO D23-06-I02: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # ANALYZE는 쿼리를 실행해 실행 시간과 실제 행 수를 기록하고,
    # BUFFERS는 데이터 페이지를 메모리에서 찾거나 읽은 횟수를 보여 줍니다.
    # 같은 모델의 문서만 남겨 코사인 거리가 작은 순서로 최대 3건을 조회합니다.
    return """EXPLAIN (ANALYZE, BUFFERS)
    SELECT id FROM day23_search_docs
    WHERE embedding_model=%s
    ORDER BY embedding <=> %s ASC LIMIT 3"""
# END D23-06-I02

# BEGIN D23-06-I03
def read_scan_type(plan_lines):
    """
    TODO D23-06-I03 · read_scan_type
        PostgreSQL EXPLAIN 텍스트에서 Index Scan / Seq Scan / 기타를 구분하세요.
        계획이 없으면 unknown, index가 있으면 index, seq면 seq로 반환합니다.
        작은 9건 테이블에서 Seq Scan은 오류가 아닙니다.
    """
    # TODO D23-06-I03: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 여러 줄을 공백으로 이어 붙여 스캔 이름을 찾습니다.
    # Index Scan은 인덱스를 이용하고, Seq Scan은 테이블의 행을 순서대로 읽습니다.
    # 둘 다 보이면 이 함수는 index를 우선 반환합니다.
    s=' '.join(plan_lines)
    if 'Index Scan' in s:return 'index'
    if 'Seq Scan' in s:return 'seq'
    return 'unknown'
# END D23-06-I03

# BEGIN D23-06-I04
def check_operator_compatibility(sql,operator_class):
    """
    TODO D23-06-I04 · check_operator_compatibility
        SQL이 cosine <=> 를 사용하면 vector_cosine_ops가 맞는지 검사하세요.
        불일치 시 ValueError, 정상 시 True.
        L2 / negative inner product와 혼동하지 않습니다.
    """
    # TODO D23-06-I04: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # <=>는 코사인 거리 연산자입니다. 이 연산자를 쓰는 경우에 인덱스 설정을 검사합니다.
    if '<=>' in sql and operator_class!='vector_cosine_ops':raise ValueError('operator_class_mismatch')
    return True
# END D23-06-I04

# BEGIN D23-06-I05
def compare_topk(exact_ids,ann_ids):
    """
    TODO D23-06-I05 · compare_topk
        Exact Top-k와 ANN Top-k의 ID 집합 일치율을 계산하세요.
        겹치는 ID 수 / exact_ids 수. 빈 exact_ids는 None.
        인덱스 속도나 모델 품질을 만들어 내면 안 됩니다.
    """
    # TODO D23-06-I05: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 기준 목록이 비어 있으면 비율을 계산할 수 없으므로 None을 반환합니다.
    if not exact_ids:return None
    # set은 중복을 제거하고, &는 양쪽에 공통으로 있는 ID만 남깁니다.
    # 기준 목록의 고유 ID가 분모이므로 검색 결과의 순서는 계산에 영향을 주지 않습니다.
    return len(set(exact_ids)&set(ann_ids))/len(set(exact_ids))
# END D23-06-I05
