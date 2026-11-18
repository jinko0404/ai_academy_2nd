# Day23 / 실습 01 / 응용 / mentee
# main.py에서 받은 지식 행을 선택하고, E5 문서 입력을 만들고,
# 원문과 연결할 수 있도록 기록한 뒤 ID와 필수 필드를 검사합니다.


# BEGIN D23-01-I01
def select_original(rows):
    """
    TODO D23-01-I01 · select_original
        기존 수업에서 사용한 101·102·103만 ID 오름차순으로 선택합니다.
        입력: 전체 지식 목록(list[dict]). 출력: 원래 지식 3개를 담은 list.
        원본 딕셔너리의 id/title/content를 수정하지 마세요.
        개념: 원문 ID 고정·검색할 자료의 범위.
    """
    # TODO D23-01-I01: 위 입력·출력 계약에 맞게 직접 구현합니다.
    selected_rows = []
    for row in rows:
        if row["id"] in (101, 102, 103):
            # 선택한 행을 새 딕셔너리로 복사해 결과에 담습니다.
            selected_rows.append(dict(row))
    return selected_rows
# END D23-01-I01

# BEGIN D23-01-I02
def make_passage_input(row):
    """
    TODO D23-01-I02 · make_passage_input
        E5 문서 입력 텍스트를 "passage: 제목 본문" 형식으로 만드세요.
        입력: 지식 한 행. 출력: 문자열. 제목과 본문을 공백 하나로 구분하세요.
        개념: FAQ 질문과 달리 지식 본문을 검색 대상으로 사용합니다.
    """
    # TODO D23-01-I02: 위 입력·출력 계약에 맞게 직접 구현합니다.
    title = row["title"]
    content = row["content"]

    # passage:는 E5에 문서 입력임을 알려주고, 가운데 공백은 제목과 본문을 구분합니다.
    passage = "passage: " + title + " " + content

    return passage
# END D23-01-I02

# BEGIN D23-01-I03
def make_mapping(rows):
    """
    TODO D23-01-I03 · make_mapping
        각 지식의 id/title/content/category/source_version 및 passage_input을 기록하세요.
        입력: 선택된 지식 목록. 출력: 딕셔너리 목록.
        개념: 벡터를 생성하기 전에도 원문·ID·버전을 추적해야 합니다.
    """
    # TODO D23-01-I03: 위 입력·출력 계약에 맞게 직접 구현합니다.
    result = []
    for row in rows:
        passage_input = make_passage_input(row)

        # 나중에 문서 ID와 버전으로 어떤 원문에서 만든 입력인지 찾을 수 있습니다.
        record = {
            "id" : row["id"],
            "title" : row["title"],
            "content" : row["content"],
            "category" : row["category"],
            "source_version" : row["source_version"],
            "passage_input" : passage_input
        }

        result.append(record)
    
    return result
# END D23-01-I03

# BEGIN D23-01-I04
def reject_duplicate_ids(rows):
    """
    TODO D23-01-I04 · reject_duplicate_ids
        모든 9개 지식의 ID 중복 여부를 점검하고 중복이면 ValueError를 내세요.
        단순 len만 비교하지 말고 id의 타입이 int(True 제외)인지도 확인합니다.
        출력: 중복이 없으면 실제 ID 집합(set).
    """
    # TODO D23-01-I04: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # get은 id 키가 빠진 행에서 None을 돌려주므로 아래 검사에서 함께 걸러집니다.
    ids=[r.get('id') for r in rows]
    # bool은 int의 하위 타입이므로 type(...) is int로 엄격하게 검사합니다.
    if any(type(x) is not int or x<=0 for x in ids):raise ValueError('id 타입 오류')
    # set은 같은 값을 하나만 담으므로, 길이가 줄었다면 중복 ID가 있었다는 뜻입니다.
    if len(ids)!=len(set(ids)):raise ValueError('duplicate_id')
    return set(ids)
# END D23-01-I04

# BEGIN D23-01-I05
def reject_missing_fields(rows):
    """
    TODO D23-01-I05 · reject_missing_fields
        title/content/category/source_version의 누락 또는 공백 문자열을 거부하세요.
        입력: 전체 지식 목록. 정상 출력: True. 잘못된 항목은 ValueError.
        개념: 잘못된 원문은 정상적인 Embedding 입력이 될 수 없습니다.
    """
    # TODO D23-01-I05: 위 입력·출력 계약에 맞게 직접 구현합니다.
    for r in rows:
        for name in ('title','content','category','source_version'):
            # 누락되었거나 문자열이 아니거나, 양쪽 공백을 없애면 빈 문자열인 경우를 거부합니다.
            if not isinstance(r.get(name),str) or not r[name].strip():
                raise ValueError('missing_or_blank:'+name)
    return True
# END D23-01-I05
