# 전체 흐름: 벡터 좌표 준비 → 방향 유사도 계산 → 점수 해석 → 그림과 JSON 저장.
# 임베딩은 데이터를 숫자 벡터로 표현하는 것이다.
# 여기서는 모델로 문장을 변환하지 않고, 직접 정한 2차원 벡터로 개념만 확인한다.

# Path는 파일과 폴더의 경로를 다루는 도구이다.
from pathlib import Path
# sys는 파이썬 실행 환경을 다룬다. 여기서는 import 검색 경로를 설정한다.
import sys

# __file__은 현재 main.py 경로, resolve()는 절대 경로, parent는 상위 폴더이다.
# HERE는 현재 파일이 들어 있는 03_embeddings 폴더가 된다.
HERE = Path(__file__).resolve().parent
# 한 단계 위인 01_basic 폴더. 이 안에 common 폴더가 있다.
LEVEL_ROOT = HERE.parent
# 모듈 검색 목록 맨 앞에 경로 문자열을 추가해 common을 찾도록 한다.
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# torch는 숫자 배열인 텐서(tensor)를 만들고 계산하는 PyTorch 라이브러리이다.
import torch
# F는 torch.nn.functional의 짧은 별명이다. 이 파일에서는 직접 사용하지 않는다.
import torch.nn.functional as F
# matplotlib은 그래프를 그리는 라이브러리이다.
import matplotlib
# Agg는 별도 그래프 창을 띄우지 않고 이미지 파일로 저장할 때 쓰는 방식이다.
# pyplot을 가져오기 전에 설정한다.
matplotlib.use("Agg")
# pyplot을 plt라는 짧은 이름으로 사용한다.
import matplotlib.pyplot as plt
# cosine은 방향 유사도 계산, write_json은 JSON 파일 저장을 담당하는 공통 함수이다.
from common.support import cosine, write_json

# 난수의 시작값을 고정한다. 다만 이 파일의 좌표와 계산에는 난수를 쓰지 않는다.
torch.manual_seed(42)
# PyTorch의 CPU 연산에 사용할 스레드 수를 1로 설정한다.
torch.set_num_threads(1)

# 아래 들여쓴 코드들을 main 함수로 묶는다. 맨 아래 main()에서 호출한다.
def main():
    # PPT 33장과 같은 교육용 2D 벡터. SentenceTransformer가 만든 실제 의미 벡터가 아닙니다.
    # 딕셔너리 {이름: 좌표 리스트}이다. 각 리스트의 두 숫자는 [x, y]이다.
    # A=(1, 0): 원점에서 오른쪽, B=(0.8, 0.2): 오른쪽에서 조금 위쪽,
    # C=(-1, 0): 왼쪽을 향한다. 벡터의 방향은 원점을 기준으로 생각한다.
    # coords["A"]처럼 이름을 키로 사용하면 [1.0, 0.0]을 꺼낼 수 있다.
    coords = {"A":[1.0,0.0],"B":[0.8,0.2],"C":[-1.0,0.0]}

    # TODO D20-03-B01: A-B와 A-C 벡터의 Cosine Similarity를 계산하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-B01
    # cosine(첫 벡터, 두 번째 벡터)는 두 벡터 방향이 얼마나 비슷한지 계산한다.
    # 공통 함수 안에서 리스트를 float32 텐서로 바꾸어 계산한 뒤 float으로 돌려준다.
    # 0이 아닌 벡터끼리 비교할 때: 1은 같은 방향, 0은 직각, -1은 반대 방향이다.
    # 계산식: 두 벡터의 내적 / (첫 벡터 길이 × 두 번째 벡터 길이).
    # 내적은 같은 위치의 숫자끼리 곱한 뒤 더한 값이다.
    # A·B = 1×0.8 + 0×0.2 = 0.8.
    # A의 길이는 1, B의 길이는 sqrt(0.8² + 0.2²)이므로 결과는 약 0.9701이다.
    s_ab = cosine(coords["A"], coords["B"])  # TODO: cosine(coords["A"], coords["B"])
    # A와 C는 정확히 반대 방향이므로 -1.0이다.
    # s_ab와 s_ac는 각각 계산 결과 숫자 하나를 담는 변수이다.
    s_ac = cosine(coords["A"], coords["C"])  # TODO: cosine(coords["A"], coords["C"])
    # END D20-03-B01

    # TODO D20-03-B02: 점수의 부호·범위를 보고 어느 벡터 방향이 유사한지 판별하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-B02
    # 점수의 의미를 문장으로 기록할 딕셔너리이다.
    # "참일 때 값 if 조건 else 거짓일 때 값"은 조건에 따라 값 하나를 고른다.
    # 0.9와 -0.99는 이 예제에서 정한 판단 기준이다.
    # A_B 부분을 풀어 쓰면:
    # if s_ab > 0.9:
    #     description = "방향이 비슷함"
    # else:
    #     description = "차이 있음"
    # interpretation["A_B"] = description
    interpretation = {
        # 약 0.9701 > 0.9는 참이므로 "방향이 비슷함"이 들어간다.
        "A_B": "방향이 비슷함" if s_ab > 0.9 else "차이 있음",  # TODO: s_ab의 방향 해석 (힌트: 1에 가까운가?)
        # -1.0 < -0.99는 참이므로 "반대 방향"이 들어간다. A-C이므로 s_ac를 쓴다.
        "A_C": "반대 방향" if s_ac < -0.99 else "차이 있음",  # TODO: s_ac의 방향 해석 (힌트: -1에 가까운가?)
        # 실제 임베딩에서도 방향 유사도만으로 답변 내용이 사실인지 판정할 수는 없다.
        "note": "유사도 점수만으로 답변의 사실성을 보장하지 않는다",
    }
    # END D20-03-B02
    # 유사도 순위를 기록할 자리이다. 기본 실습에서는 순위를 계산하지 않아 비워 둔다.
    ranked = []
    # coords.values()는 이름을 제외한 좌표들만 꺼낸다.
    # list(...) 결과: [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]].
    # torch.tensor(...)는 이 중첩 리스트를 계산용 텐서로 바꾼다.
    # dtype=torch.float32는 숫자를 32비트 실수로 저장하라는 뜻이다.
    # 벡터 3개를 행으로 쌓았으므로 행렬(matrix)은 3행 2열이다.
    matrix=torch.tensor(list(coords.values()),dtype=torch.float32)
    # shape는 데이터의 크기 구조이다. 실제 좌표 값이 아니라 각 차원의 길이를 알려준다.
    # matrix[0]은 첫 행인 A 벡터이고 shape는 (2,)이다. 숫자가 2개라는 뜻이다.
    # matrix 전체의 shape는 (3, 2)이다. 벡터 3개, 벡터당 숫자 2개라는 뜻이다.
    # list(...)로 바꿔 JSON에 저장할 결과는 {"vector": [2], "matrix": [3, 2]}이다.
    shapes={"vector":list(matrix[0].shape),"matrix":list(matrix.shape)}
    # 토큰별 벡터와 이를 모은 벡터의 크기를 기록할 후속 실습용 자리이다.
    # 이 파일에서는 해당 계산을 하지 않으므로 빈 리스트로 둔다.
    token_shape = []
    pooled_shape = []
    # 이 실습 결과로 무엇을 주장할 수 없는지 설명하는 문자열 리스트이다.
    limitations = ["이 숫자는 예시 벡터이며 문장 의미 품질을 증명하지 않는다"]

    # TODO D20-03-B03: 2D 점 그래프와 cosine_report.json을 outputs에 저장하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-B03
    # fig는 그림 전체, ax는 그 안에서 점과 축을 그릴 영역이다.
    # subplots가 돌려준 값 두 개를 fig와 ax에 나눠 받는다.
    # figsize=(6, 4)는 그림 크기를 가로 6인치, 세로 4인치로 정한다.
    fig, ax = plt.subplots(figsize=(6, 4))
    # coords.items()는 (이름, 좌표) 쌍을 하나씩 꺼낸다.
    # 첫 반복: ("A", [1.0, 0.0]) → label="A", x=1.0, y=0.0.
    # label, (x, y)는 바깥 묶음뿐 아니라 안쪽 좌표도 각각 변수에 나눠 받는 문법이다.
    # 풀어 쓰면:
    # for label, point in coords.items():
    #     x = point[0]
    #     y = point[1]
    for label, (x, y) in coords.items():
        # scatter는 (x, y) 위치에 점을 그린다. 벡터 화살표를 그리는 코드는 아니다.
        ax.scatter(x, y)  # 각 벡터를 점으로 표시
        # annotate는 점 옆에 label 문자열(A/B/C)을 적는다.
        # (x, y)는 기준점, xytext=(5, 5)는 글자를 오른쪽과 위로 옮길 양이다.
        # textcoords="offset points"는 그 이동량의 단위를 포인트로 정한다.
        # 즉, 데이터 좌표를 5씩 더하는 것이 아니라 화면의 표시 위치를 조금 옮긴다.
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
        # TODO: ax.annotate(...)로 점 옆에 A/B/C 이름 표시
    # 가로축 이름, 세로축 이름, 그림 제목을 설정한다.
    ax.set(xlabel="x1", ylabel="x2", title="Educational 2D vectors")
    # 격자선을 표시한다. alpha=0.2는 투명도를 조절해 연하게 보이게 한다.
    ax.grid(alpha=0.2)
    # 제목과 축 이름 등이 그림 밖으로 잘리지 않도록 여백을 자동 조정한다.
    fig.tight_layout()
    # Path의 /는 여기서 경로를 이어 붙인다. 현재 폴더 아래 outputs를 만든다.
    # exist_ok=True이므로 outputs 폴더가 이미 있어도 오류를 내지 않는다.
    (HERE / "outputs").mkdir(exist_ok=True)
    # 그림 전체를 outputs/vectors.png 파일로 저장한다.
    fig.savefig(HERE / "outputs" / "vectors.png")
    # 저장을 끝낸 그림을 닫아 관련 자원을 해제한다. 저장된 파일은 그대로 남는다.
    plt.close(fig)
    # 계산 결과를 딕셔너리로 묶어 outputs/cosine_report.json에 저장한다.
    # write_json은 파이썬 데이터를 JSON 문자열로 변환해 UTF-8 파일로 쓴다.
    # "vectors": coords처럼 왼쪽은 저장할 항목 이름, 오른쪽은 그 항목의 값이다.
    write_json(HERE / "outputs" / "cosine_report.json", {
        # 원본 A/B/C 좌표.
        "vectors": coords,
        # 계산한 유사도 숫자 두 개를 담은 내부 딕셔너리.
        "cosine": {"AB": s_ab, "AC": s_ac},  # TODO: 계산한 s_ab와 s_ac
        # 점수를 해석한 문장들.
        "interpretation": interpretation,
        # 순위 결과 자리. 기본 실습에서는 []로 저장된다.
        "ranked": ranked,
        # 벡터와 행렬의 크기 정보.
        "shapes": shapes,
        # 기본 실습에서는 두 항목 모두 []로 저장된다.
        "token_shape": token_shape,
        "pooled_shape": pooled_shape,
        # 예제 결과를 해석할 때의 한계 설명.
        "limitations": limitations,
    })
    # END D20-03-B03

# 이 파일을 직접 실행하면 __name__이 "__main__"이 되어 아래 함수를 호출한다.
# 다른 파일에서 import하면 main 함수는 자동 실행되지 않는다.
if __name__=="__main__":
    main()
