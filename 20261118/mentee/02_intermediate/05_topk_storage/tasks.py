# Day23 / 실습 05 / 응용 / mentee
# main.py에서 사용할 저장·검색용 SQL을 제공하고, 문서 입력과 검색 옵션을 준비합니다.
from day23kit.data import passage_text
from day23kit.db import TABLE_SQL, SQL_TOPK, SQL_CATEGORY

# BEGIN D23-05-I01
def schema_sql():
    """
    TODO D23-05-I01 · schema_sql
        기존 knowledge 테이블을 건드리지 않는 day23_search_docs의 CREATE TABLE SQL을 작성하세요.
        필드: id(PK),title,content,category,source_version,embedding_model,embedding vector(384).
        DROP/TRUNCATE를 사용하지 않습니다.
    """
    # TODO D23-05-I01: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 별도 검색 테이블에 원문과 384차원 벡터를 함께 저장하는 CREATE TABLE 문입니다.
    # id를 기본 키로 써서 문서 한 건을 식별합니다.
    return TABLE_SQL
# END D23-05-I01

# BEGIN D23-05-I02
def prepare_document_rows(rows):
    """
    TODO D23-05-I02 · prepare_document_rows
        지식 101~109의 ID/제목/본문/분류/버전 및 passage 입력을 순서대로 묶으세요.
        반환: [{'id', 'passage_input', ...}, ...]. 같은 ID를 새로 만들지 않습니다.
    """
    # TODO D23-05-I02: 위 입력·출력 계약에 맞게 직접 구현합니다.
    result = []
    for row in rows:
        # 원본 행을 복사하고, 복사본에만 'passage: 제목 본문' 문자열을 추가합니다.
        record = dict(row)
        record["passage_input"] = passage_text(row)
        # 입력 순서대로 추가하므로 결과의 문서 순서도 같습니다.
        result.append(record)

    return result 
# END D23-05-I02

# BEGIN D23-05-I03
def topk_sql():
    """
    TODO D23-05-I03 · topk_sql
        Cosine distance(<=>) 오름차순, k개 제한 SQL을 작성하세요.
        SELECT에 id,title,content,category,source_version,embedding_model,distance 포함.
        E5 외 다른 공간 혼합을 막도록 embedding_model을 필터합니다.
    """
    # TODO D23-05-I03: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # <=>로 코사인 거리를 구해 작은 순서(ASC)로 정렬하고, LIMIT으로 개수를 제한합니다.
    # embedding_model 조건으로 질문 벡터와 같은 모델에서 만든 문서 벡터만 비교합니다.
    return SQL_TOPK
# END D23-05-I03

# BEGIN D23-05-I04
def filtered_sql():
    """
    TODO D23-05-I04 · filtered_sql
        category 필터가 포함된 Top-k SQL을 작성하세요.
        쿼리 벡터와 category, 모델, k는 문자열 합치기가 아닌 %s 바인딩입니다.
    """
    # TODO D23-05-I04: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 일반 Top-k 검색에 category 조건을 더합니다.
    # %s는 값이 들어갈 자리입니다. 실행할 때 SQL과 실제 값을 따로 전달합니다.
    return SQL_CATEGORY
# END D23-05-I04

# BEGIN D23-05-I05
def check_search_options(k,category):
    """
    TODO D23-05-I05 · check_search_options
        k는 bool을 제외한 정수 1~9, category는 None 또는 course/development/support인지 검사하세요.
        잘못된 값은 ValueError. 정상 시 {'k':k,'category':category} 반환.
    """
    # TODO D23-05-I05: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 타입부터 확인하면 문자열이나 bool을 숫자 범위와 비교하기 전에 거를 수 있습니다.
    if type(k) is not int or not (k >= 1 and k <= 9):
        raise ValueError('invalid_k')
    # None은 분류 전체를 뜻합니다. 값이 있다면 허용된 분류 중 하나여야 합니다.
    if category is not None and category not in ("course", "development", "support"):
        raise ValueError('invalid_category')

    return {'k':k,'category':category}
# END D23-05-I05
