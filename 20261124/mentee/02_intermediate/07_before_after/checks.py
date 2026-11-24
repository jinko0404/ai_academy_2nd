"""단계별 역할 분리: checks.py / 독립 실행은 main.py. """

# 재정렬 전후를 비교한 뒤, 어떤 질문을 썼고 두 문서 목록이 어떤 관계인지 기록합니다.
# 결과는 저장된 JSON의 row.controlled_conditions에서 확인할 수 있습니다.

def check_conditions(case, before, ranked):
    """Gold Label은 검색과 Reranker의 입력이 아니라 평가할 때만 사용합니다."""
    # BEGIN D27-07-I03
    # TODO D27-07-I03: Gold Label은 검색과 Reranker의 입력이 아니라 평가할 때만 사용합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # case에는 실습 질문 정보가 있습니다. 결과에 질문 번호(Q01)와 문서 분류(시설)를 남깁니다.
    # 여기의 before와 ranked는 Document 객체 목록입니다. metadata는 각 문서의 부가 정보입니다.
    # gold_separate는 처음 검색된 문서의 metadata에 'gold'라는 항목이 없는지 하나씩 확인합니다.
    # same_candidates는 재정렬 결과에 나온 ID가 모두 처음 후보에 있었는지 확인합니다.
    # 예를 들어 처음 후보 [A, B, C]에서 결과가 [C, A]이면 통과하고 [C, D]이면 실패합니다.
    return {'question_id':case['id'],'category':case['category'],
            'gold_separate':all('gold' not in d.metadata for d in before),
            'same_candidates':set(d.metadata['chunk_id'] for d in ranked).issubset(
                set(d.metadata['chunk_id'] for d in before))}
    # END D27-07-I03
