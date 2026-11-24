"""단계별 역할 분리: steps.py / 독립 실행은 main.py. """

# 첫 함수는 문서 순위를 비교해 결과 딕셔너리를 만들고, 둘째 함수는 그 결과를 한 문장으로 설명합니다.

def compare_rankings(before, after, gold):
    """고정한 관련 문서 ID를 Top-2에 포함했는지 확인합니다."""
    # BEGIN D27-07-I01
    # TODO D27-07-I01: 고정한 관련 문서 ID를 Top-2에 포함했는지 확인합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # before와 after는 각각 재정렬 전과 후의 문서 ID 목록입니다. 각 목록의 첫 ID가 1위입니다.
    # gold는 이 질문에 관련 있다고 미리 표시해 둔 문서 ID 목록입니다. 비교할 때 사용합니다.
    # compare_lists()는 각 목록의 앞 두 ID 중 gold와 겹치는 ID가 하나라도 있는지 확인합니다.
    # 이번 교육용 예시에서 관련 문서 FAC-102-C1은 재정렬 전 3위, 재정렬 후 1위입니다.
    # 그래서 before_contains_gold는 False, after_contains_gold는 True가 됩니다.
    from coursekit.evaluation import compare_lists
    return compare_lists(before,after,gold,k=2)
    # END D27-07-I01


def interpret_change(report):
    """순위가 바뀌어도 반드시 개선됐다는 결론을 내리지 않습니다."""
    # BEGIN D27-07-I02
    # TODO D27-07-I02: 순위가 바뀌어도 반드시 개선됐다는 결론을 내리지 않습니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # report에는 첫 함수의 비교 결과가 들어옵니다. 여기서 before와 after는 문서 목록 대신
    # 각각 '재정렬 전과 후의 상위 2개 안에 관련 문서가 있었는가'를 나타냅니다.
    # 값이 None이면 미리 표시한 관련 문서가 없는 경우입니다. 두 값이 같으면 포함 여부는 그대로입니다.
    # False → True와 True → False 모두 포함 여부가 달라진 경우입니다. 이 문장은 한 질문의 결과만 설명합니다.
    before=report['before_contains_gold'];after=report['after_contains_gold']
    if before is None:return 'gold 없음: 인간 검토 필요'
    if before==after:return 'Top-k 관련 문서 포함 여부는 변하지 않음'
    return '관련 문서 Top-k 포함 여부가 달라짐 (추가 검증 필요)' 
    # END D27-07-I02
