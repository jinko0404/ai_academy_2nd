# Day23 / 실습 08 / 응용 / mentee
# main.py에서 호출하는 함수들입니다. 입력 검사, 후보 검색, 결과 정리, 실행 기록을 담당합니다.
from day23kit.fixtures import fixture_search
from day23kit.embedding import get_encoder, encode_question
from day23kit.db import connect_db, exact_topk, check_data_ready

# BEGIN D23-08-I01
def validate_question(question):
    """
    TODO D23-08-I01 · validate_question
        질문은 1자 이상의 문자열이어야 하며 앞뒤 공백을 제거하세요.
        빈 값/공백만 있는 값이면 ValueError. 반환은 정리된 질문.
        API가 연결되어 있지 않아도 검사 가능한 순수 함수입니다.
    """
    # TODO D23-08-I01: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 문자열인지 먼저 확인합니다. strip()으로 공백을 제거한 뒤 비어 있어도 오류를 냅니다.
    if not isinstance(question,str) or not question.strip():raise ValueError('empty_question')
    return question.strip()
# END D23-08-I01

# BEGIN D23-08-I02
def retrieve_candidates(question,mode,k,category,env_file):
    """
    TODO D23-08-I02 · retrieve_candidates
        fixture면 교육용 fixture_search, live면 E5 query Embedding → pgvector Exact SQL을 사용하세요.
        실제 live는 exercise05에서 9개 자료를 먼저 저장해야 합니다.
        DB 연결 실패를 빈 결과 []로 바꾸지 말고 예외를 전달하세요.
    """
    # TODO D23-08-I02: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # fixture에서 바로 반환하므로 아래의 실제 모델 로드와 DB 연결을 건너뜁니다.
    if mode=='fixture':return fixture_search(question,k,category)
    encoder=get_encoder('live')
    # 공용 함수가 질문에 'query: '를 붙이고 E5 검색용 벡터로 바꿉니다.
    qvec=encode_question(encoder,question)
    # 실제 연결은 명시적으로 --mode live를 지정한 경우에만 만듭니다.
    with connect_db(env_file) as conn:
        # 실습 05의 문서가 준비됐는지 확인한 뒤, 코사인 거리가 작은 순서로 최대 k건을 찾습니다.
        # DB 연결이나 조회에서 예외가 발생하면 호출한 쪽으로 전달됩니다.
        check_data_ready(conn)
        return exact_topk(conn,qvec,k,category)
# END D23-08-I02

# BEGIN D23-08-I03
def format_candidates(hits):
    """
    TODO D23-08-I03 · format_candidates
        후보의 id/title/content/category/source_version/distance를 반환하세요.
        검색된 원문을 요약·변조하지 않고, distance는 숫자 그대로 둡니다.
        답변 상태(answered)를 자동으로 부여하지 마세요.
    """
    # TODO D23-08-I03: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 리스트 컴프리헨션으로 후보 h마다 새 딕셔너리를 하나씩 만들며 검색 순서를 유지합니다.
    # 본문과 버전은 받은 그대로 담고, 코사인 거리는 숫자로 유지하면서 소수점 여섯 자리로 반올림합니다.
    return [{'id':h['id'],'title':h['title'],'content':h['content'],
             'category':h['category'],'source_version':h['source_version'],
             'distance':round(float(h['distance']),6)} for h in hits]
# END D23-08-I03

# BEGIN D23-08-I04
def validate_options(k,category):
    """
    TODO D23-08-I04 · validate_options
        k는 bool 제외 정수 1~9, category는 None 또는 course/development/support.
        위반하면 ValueError. 정상 출력은 {'k': k,'category': category}.
        입력 검사를 끝내기 전 모델·DB를 호출하지 마세요.
    """
    # TODO D23-08-I04: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # type(k)로 True와 False를 거르고, 정수인 경우에만 1~9 범위를 확인합니다.
    # category가 None이면 분류 전체를 대상으로 검색합니다.
    if type(k) is not int or not 1<=k<=9:raise ValueError('invalid_k')
    if category not in (None,'course','development','support'):raise ValueError('invalid_category')
    return {'k':k,'category':category}
# END D23-08-I04

# BEGIN D23-08-I05
def make_run_report(question,hits,mode,elapsed_ms):
    """
    TODO D23-08-I05 · make_run_report
        run_status=candidates 또는 empty를 구분하세요.
        source_mode=(fixture:simulated, live:pgvector_exact), elapsed_ms,
        retrieved_ids, human_evidence_ok=None을 기록합니다.
        검색된 것만으로 답변 정확도·근거 승인 True를 만들지 마세요.
    """
    # TODO D23-08-I05: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # candidates는 후보가 있음을, empty는 정상 검색에서 후보가 0건임을 뜻합니다.
    # 검색된 ID와 경과 시간을 기록합니다. 근거 승인 여부는 사람이 검토하기 전까지 미정입니다.
    return {'run_status':'candidates' if hits else 'empty',
            'source_mode':'simulated' if mode=='fixture' else 'pgvector_exact',
            'retrieved_ids':[h['id'] for h in hits],
            'elapsed_ms':round(elapsed_ms,3),
            'human_evidence_ok':None}
# END D23-08-I05
