# Day23 / 실습 02 / 응용 / mentee
# main.py의 예제 벡터를 받아 유사도와 거리, 단위 벡터, 후보 순위를 계산합니다.
import numpy as np

# BEGIN D23-02-I01
def dot(a,b): 
    """
    TODO D23-02-I01 · dot
        같은 위치의 숫자를 곱해 합산하는 내적을 구현하세요.
        예: (1,0)·(0.8,0.6)=0.8. 입력 NumPy 배열 2개, 출력 float.
    """
    # TODO D23-02-I01: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 같은 위치의 성분을 곱해 더하고, NumPy 결과를 파이썬 float로 변환합니다.
    return float(np.dot(a, b))
# END D23-02-I01

# BEGIN D23-02-I02
def cosine(a,b): # 두 벡터 사이의 코사인 유사도를 구하는 함수
    """
    TODO D23-02-I02 · cosine
        내적을 두 벡터 길이의 곱으로 나누어 Cosine similarity를 구하세요.
        q=(1,0), A=(0.8,0.6)라면 0.8입니다.
        입력: NumPy 배열 2개. 출력: float. 영벡터는 무조건 거부합니다.
    """
    # TODO D23-02-I02: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # norm은 벡터의 길이입니다. 길이가 1e-12 이하이면 영벡터로 취급합니다.
    na=float(np.linalg.norm(a))
    nb=float(np.linalg.norm(b))

    if na<=1e-12 or nb<=1e-12:
        raise ValueError('zero_vector')

    # 내적을 두 길이의 곱으로 나누면 벡터 크기의 영향이 사라지고 방향을 비교할 수 있습니다.
    return dot(a=a, b=b) / (na * nb)
# END D23-02-I02

# BEGIN D23-02-I03
def l2(a,b): # 두 벡터 사이 거리를 계산하는 함수
    """
    TODO D23-02-I03 · l2
        두 벡터의 좌표 차이로 L2 거리를 구하세요. 출력 float.
        q=(1,0), A=(0.8,0.6) → sqrt(0.4) ≈ 0.6325.
    """
    # TODO D23-02-I03: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # a - b로 좌표별 차이를 구하고 그 벡터의 길이를 재면 두 점 사이의 직선 거리입니다.
    return float(np.linalg.norm(a - b));
# END D23-02-I03

# BEGIN D23-02-I04
def normalize(a):
    """
    TODO D23-02-I04 · normalize
        임의 벡터 a를 길이가 1인 단위 벡터로 변환하세요.
        기존 배열을 직접 변경하지 않습니다. 영벡터는 ValueError.
    """
    # TODO D23-02-I04: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 각 성분을 같은 길이 n으로 나누면 방향을 유지하면서 길이가 1인 새 배열이 됩니다.
    n = float(np.linalg.norm(a))
    if n<=1e-12:
        raise ValueError('zero_vector')

    return a/n
# END D23-02-I04

# BEGIN D23-02-I05
def rank_dot_vs_cosine(q,candidates):
    """
    TODO D23-02-I05 · rank_dot_vs_cosine
        후보 {이름:벡터}를 Dot 내림차순과 Cosine 내림차순으로 각각 정렬하세요.
        결과: {'dot':[이름...], 'cosine':[이름...]}.
        크기가 큰 벡터가 Dot 순위를 바꿀 수 있음을 확인합니다.
    """
    # TODO D23-02-I05: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # sorted는 딕셔너리의 키인 후보 이름을 순회합니다. n은 현재 후보의 이름입니다.
    # (-점수, 이름)을 기준으로 오름차순 정렬하면 높은 점수가 먼저 오고,
    # 점수가 같으면 이름 오름차순으로 정렬됩니다.
    d=sorted(candidates,key=lambda n:(-dot(q,candidates[n]),n))
    c=sorted(candidates,key=lambda n:(-cosine(q,candidates[n]),n))
    return {'dot':d,'cosine':c}
# END D23-02-I05
