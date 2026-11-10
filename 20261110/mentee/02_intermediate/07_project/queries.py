"""
[Day17 - A07 기본 개념: 조회 계산의 분리]

1. 역할 분리
   - 전달받은 목록에서 결과만 구하며 파일이나 HTTP 응답을 다루지 않음
   - main.py가 데이터를 준비하고 None을 404로 바꾸는 역할을 맡음

2. 선택 인자와 리스트 컴프리헨션
   - str | None은 문자열 또는 None을 허용한다는 타입 표기임
   - [값 for 항목 in 목록 if 조건]은 조건에 맞는 값만 새 목록으로 만듦
   - category 비교는 대소문자까지 정확히 일치해야 함

3. 단일 항목 조회
   - dict | None은 항목 하나 또는 찾지 못했다는 None을 반환함
   - return은 일치 항목을 찾는 즉시 함수를 종료함
"""

# category가 없으면 전체, 있으면 해당 분류의 목록을 반환
def select_category(entries: list[dict], category: str | None) -> list[dict]:
    if category is None:
        # 새 파일을 읽지 않고 전달받은 목록 자체를 반환
        return entries
    # 일치 항목이 없으면 빈 목록이며 이는 오류가 아님
    return [entry for entry in entries if entry["category"] == category]


def find_entry(entries: list[dict], knowledge_id: int) -> dict | None:
    # 각 지식의 ID를 확인하며 목록 위치와 ID를 혼동하지 않음
    for entry in entries:
        if entry["id"] == knowledge_id:
            # 목록이 아닌 지식 dict 한 개를 반환
            return entry
    # 반복을 끝내도 찾지 못했을 때만 실행
    return None
