"""단계별 역할 분리: checks.py / 독립 실행은 main.py. """

# 체인이 실행된 뒤 반환한 결과에서 trace(실행 기록)를 검사합니다.
# 여기서 만든 세 가지 참·거짓 값은 결과 JSON의 row.trace_check에 기록됩니다.

def validate_trace(result):
    """원 질문·검색용 질문·후보·Context ID를 각각 확인합니다."""
    # BEGIN D27-06-I03
    # TODO D27-06-I03: 원 질문·검색용 질문·후보·Context ID를 각각 확인합니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # 1. result['trace']는 어떤 문장으로 검색했고 어떤 문서를 골랐는지 적어 둔 기록입니다.
    #    아래에서는 이 기록을 짧게 t라고 부릅니다.
    # 2. questions_separate는 질문 두 칸에 값이 있는지 봅니다. 이번 오프라인 실행에서는
    #    원 질문 칸에 '그럼 취소는 어떻게 해?', 검색어 칸에 '회의실 예약을 취소하는 방법은?'이 들어 있습니다.
    #    두 문장이 같은 경우에도 두 칸에 값이 있으면 통과합니다.
    # 3. ids_recorded는 두 종류의 문서 ID가 모두 리스트인지 봅니다. 이번 실행에서 candidate_ids는
    #    검색한 문서 5개의 ID, included_ids는 근거로 넘긴 두 문서의 ID를 담고 있습니다.
    # 4. issubset()은 왼쪽의 ID가 오른쪽에 전부 들어 있는지 봅니다. 이번 실행에서 근거로 고른
    #    FAC-102-C1과 FAC-103-C1이 모두 검색 후보 목록에 있으므로 included_from_candidates는 True입니다.
    t=result['trace'];return {'questions_separate':bool(t['original_question'] and t['search_query']),
            'ids_recorded':isinstance(t['candidate_ids'],list) and isinstance(t['included_ids'],list),
            'included_from_candidates':set(t['included_ids']).issubset(t['candidate_ids'])}
    # END D27-06-I03
