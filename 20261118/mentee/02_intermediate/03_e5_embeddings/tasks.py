# Day23 / 실습 03 / 응용 / mentee
# 텍스트를 준비하고, 인코더로 벡터를 만들고, 그 결과를 검증하는 함수들입니다.
import numpy as np
from day23kit.data import passage_text
from day23kit.embedding import validate_vectors

# BEGIN D23-03-I01
def make_passages(rows):
    """
    TODO D23-03-I01 · make_passages
        지식 목록을 E5 문서 입력 문자열 목록으로 변환하세요.
        문서에는 'passage: ' + 제목 + ' ' + 본문을 사용합니다.
        한 행마다 정확히 한 문자열을 만들어 순서를 유지하세요.
    """
    # TODO D23-03-I01: 위 입력·출력 계약에 맞게 직접 구현합니다.
    passages =[]
    for row in rows:
        # 제목과 본문으로 만든 문자열을 뒤에 붙이면 입력 문서의 순서가 유지됩니다.
        passages.append(passage_text(row))
    return passages
# END D23-03-I01

# BEGIN D23-03-I02
def make_queries(questions):
    """
    TODO D23-03-I02 · make_queries
        질문 목록 각 요소 앞에 query: 공백을 붙이세요.
        출력: 입력과 동일한 순서의 문자열 리스트.
        빈 질문을 추가해 그럴듯한 벡터를 만들지 마세요.
    """
    # TODO D23-03-I02: 위 입력·출력 계약에 맞게 직접 구현합니다.
    queries = []

    for question in questions:
        # question은 질문 문자열 하나입니다. query:와 공백을 앞에 붙입니다.
        queries.append("query: " + question)

    return queries
# END D23-03-I02

# BEGIN D23-03-I03
def encode_both(encoder,passages,queries):
    """
    TODO D23-03-I03 · encode_both
        같은 Encoder에 문서와 질문을 나누어 encode하세요.
        반드시 normalize_embeddings=True, convert_to_numpy=True를 전달합니다.
        결과: (doc_vectors, query_vectors). 기본필수의 모의 모드와
        --mode live의 E5 실측을 구분하여 같은 API를 사용합니다.
    """
    # TODO D23-03-I03: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 같은 인코더를 두 번 호출해 문서와 질문의 벡터를 각각 얻습니다.
    # normalize_embeddings=True는 각 벡터의 길이를 1로 맞춥니다.
    # convert_to_numpy=True는 NumPy 배열로 받는 옵션이고, batch_size=8은 문서 처리 묶음 크기입니다.
    docs=encoder.encode(passages,normalize_embeddings=True,convert_to_numpy=True,batch_size=8)
    queries_vec=encoder.encode(queries,normalize_embeddings=True,convert_to_numpy=True)
    
    # 두 배열의 숫자 자료형을 float32로 통일해 (문서 배열, 질문 배열) 순서로 반환합니다.
    return np.asarray(docs,dtype=np.float32), np.asarray(queries_vec,dtype=np.float32)
# END D23-03-I03

# BEGIN D23-03-I04
def verify_alignment(rows,vectors):
    """
    TODO D23-03-I04 · verify_alignment
        지식 행 수와 벡터 행 수가 같은지 확인하고 [(ID,행번호),...]를 반환하세요.
        ID가 중복되어 있거나 행 수가 다르면 ValueError.
        예: 지식 102는 vectors[1]과 짝입니다.
    """
    # TODO D23-03-I04: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 벡터 배열의 행 하나가 문서 하나에 해당하므로 개수가 같아야 합니다.
    if len(rows) != len(vectors):
        raise ValueError("row_count_mismatch")

    seen_ids = set()  # 앞에서 나온 문서 ID를 기억합니다.
    alignment = []
    
    # 인코더가 입력 순서를 유지하므로 문서의 인덱스가 벡터의 행 번호입니다.
    # enumerate는 0부터 시작하는 인덱스와 문서 데이터를 한 쌍으로 꺼냅니다.
    for index, row in enumerate(rows):
        doc_id = row["id"]

        if doc_id in seen_ids:
            raise ValueError("duplicate_id")

        seen_ids.add(doc_id)
        alignment.append((doc_id, index))

    return alignment
# END D23-03-I04

# BEGIN D23-03-I05
def check_shapes_vectors(docs,queries,doc_count):
    """
    TODO D23-03-I05 · check_shapes_vectors
        문서 벡터 shape=(doc_count,384), 질문 shape=(1,384)를 검사하세요.
        두 배열 모두 유한하고 길이가 1인지 검증합니다.
        정상일 때 True, 오류일 때 ValueError.
    """
    # TODO D23-03-I05: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 공용 함수가 행 수와 384차원 여부, NaN·무한대 여부, 벡터 길이를 확인합니다.
    # 문서는 doc_count행, 이 실습의 질문은 한 개이므로 1행이어야 합니다.
    
    validate_vectors(docs,doc_count)
    validate_vectors(queries,1)
    
    return True
# END D23-03-I05
