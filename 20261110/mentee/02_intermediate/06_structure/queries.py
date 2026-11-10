"""
[02_intermediate 실습 06 / queries.py 기본 개념]

1. 조회 계산 분리
   - 파일이나 HTTP에 의존하지 않고 전달받은 목록에서 원하는 결과를 계산
   - 파일 읽기는 storage.py, None을 404로 바꾸는 판단은 main.py가 담당
   - 메모리 입력으로 직접 확인하기 쉬우며 외부 파일 없이도 재사용 가능

2. select_category의 계약
   - 입력은 항목 dict의 목록과 문자열 또는 None인 category
   - None이면 원래 목록을 반환하며 문자열이면 정확히 일치하는 항목 목록 반환
   - 일치 항목이 없으면 [], 빈 문자열은 None과 다른 필터 조건

3. find_entry의 계약
   - 입력 ID와 같은 항목 하나를 찾으면 dict 반환
   - 찾지 못하면 None이며 목록이나 HTTPException을 반환하지 않음
   - for문 안에서 찾은 즉시 return하므로 첫 번째 일치 항목을 반환

4. 반환 자료형과 참조
   - list[dict]는 항목 목록, dict | None은 항목 하나 또는 없음이라는 타입 표기
   - 이 함수는 원본 목록을 직접 변경하지 않음
   - 반환 항목은 깊은 복사본이 아니므로 호출자가 수정하면 원래 객체에 영향 가능

5. 순회·조건·반환
   - for로 항목을 하나씩 확인하고 ==로 분류 또는 ID 일치 여부 판단
   - append는 결과 목록 추가, return은 결과를 넘기고 함수 종료
   - 실제 데이터의 필수 키가 존재한다는 전제이며 타입 표기만으로 런타임 검증되지는 않음
"""

"""조회 계산만 분리합니다. HTTP 응답이나 파일 저장은 여기서 처리하지 않습니다."""
# 입력 목록의 category 필터만 계산하고 결과 목록을 반환
def select_category(entries: list[dict], category: str | None) -> list[dict]:
    # 필터를 생략한 경우 전체 목록을 그대로 반환
    if category is None:
        # 원래 목록 참조를 반환하며 파일 읽기나 복사를 수행하지 않음
        return entries
    # BEGIN A06-01
    # TODO A06-01: category가 있는 경우 정확히 일치하는 항목들의 목록을 반환하세요. 이 함수에서 파일을 읽거나 HTTPException을 사용하지 마세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    # 필터에 맞는 항목을 모을 새 리스트 생성
    result_category = []
    # 전달받은 목록만 순회하며 외부 파일에는 접근하지 않음
    for item in entries:
        # 대소문자·공백을 자동 보정하지 않는 정확한 문자열 일치 검사
        if item["category"] == category:
            # 조건을 만족한 항목을 결과 목록에 추가
            result_category.append(item)

    # 없으면 빈 목록, 있으면 해당 항목들의 목록 반환
    return result_category
    # END A06-01


# ID 탐색 결과를 항목 하나 또는 None으로 반환
def find_entry(entries: list[dict], knowledge_id: int) -> dict | None:
    # BEGIN A06-02
    # TODO A06-02: entries에서 ID가 같은 항목을 찾아 반환하세요. 없으면 아래 return None으로 진행하세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    # 전달받은 목록만 순회하며 외부 파일에는 접근하지 않음
    for item in entries:
        # 항목 ID와 요청 ID 비교. 목록 인덱스와 혼동하지 않음
        if item["id"] == knowledge_id:
            # 첫 일치 항목 하나를 반환하고 탐색 종료
            return item
    # END A06-02
    # 모든 항목을 확인해도 일치하지 않으면 없음 표시. 404 처리는 호출자가 담당
    return None
