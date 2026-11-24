"""단계별 역할 분리: checks.py / 독립 실행은 main.py. """
import math

# 실습 목적
# 반환된 후보의 점수가 유한한 숫자이고 높은 점수부터 정렬됐는지 확인합니다.
# 점수는 질문과 문서의 관련도를 비교하는 데 쓰이므로 음수나 1보다 큰 값도 가능합니다.

# 전체 흐름
# steps.py가 반환한 문서 목록(ranked) → 점수 꺼내기 → 결과를 JSON의 row.score_checks에 기록

def check_rank_scores(ranked):
    """점수의 개수·유한값·정렬 순서 검사. 점수는 정답 확률이 아닙니다."""
    # BEGIN D27-05-I03
    # TODO D27-05-I03: 점수의 개수·유한값·정렬 순서 검사. 점수는 정답 확률이 아닙니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.

    # metadata는 문서의 부가 정보가 들어 있는 딕셔너리입니다. 각 문서에서
    # rerank_score 값을 하나씩 꺼내 현재 순서대로 모읍니다. 점수가 빠진 문서는 KeyError가 납니다.
    # 모델이 반환한 점수 개수와 원래 후보 수가 같은지는 재정렬 함수에서 이미 검사합니다.
    scores = [doc.metadata["rerank_score"] for doc in ranked]

    # math.isfinite()는 NaN이나 양·음의 무한대에서 False입니다. all()은 모두 참일 때 True입니다.
    # scores[1:]은 첫 점수를 건너뛴 목록입니다. zip()으로 원본과 짝지으면 이웃한 점수를 비교할 수 있습니다.
    # 예를 들어 [2.4, 1.1, 1.1]의 비교는 2.4 >= 1.1과 1.1 >= 1.1입니다. 동점도 통과합니다.
    return {
        "scores_finite": all(math.isfinite(score) for score in scores),
        "rank_sorted": all(a >= b for a, b in zip(scores, scores[1:])),
    }
    # END D27-05-I03
