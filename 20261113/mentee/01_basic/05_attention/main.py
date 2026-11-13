# 이 코드가 하는 일
# 토큰 3개가 각각 "어느 토큰의 정보를 얼마나 가져올지" 계산하는 Attention 실습이다.
# 문장을 생성하거나 모델을 학습하지 않고, 직접 정한 작은 숫자 배열로 계산한다.
#
# 전체 흐름도
# Q [3, 2] ──┐
#            ├─ Q @ K.T / sqrt(2) → scores [3, 3]: 토큰 사이의 점수
# K [3, 2] ──┘                         │
#                                      ▼
# 미래 위치 차단용 mask [3, 3] → 차단할 점수를 -inf로 바꿈
#                                      │
#                                      ▼
#                              행별 Softmax
#                                      │
#                                      ▼
#                              weights [3, 3]
#                         (각 행의 합 1, 미래 위치는 0)
#                                      │
# V [3, 2] ────────────────────────────┤
#                                      ▼
#                             weights @ V
#                                      │
#                                      ▼
#                              output [3, 2]
#                         (가져온 정보를 섞은 새 벡터)
#                                      │
#                                      ▼
#                        리스트로 변환 → JSON 저장
#
# Q(Query): 현재 위치가 다른 위치를 참조할 때 비교에 사용하는 벡터.
# K(Key): 각 참조 후보가 비교를 위해 제공하는 벡터.
# V(Value): 실제로 가져와 섞을 정보 벡터.
# 비유하면 Q와 K로 "얼마나 참고할지" 정하고, 그 비율대로 V를 섞는다.
# 실제 모델은 보통 입력 벡터를 서로 다른 변환에 통과시켜 Q/K/V를 만든다.
# 여기서는 계산을 단순화하려고 같은 숫자를 복사해서 Q/K/V로 사용한다.

# 파일과 폴더 경로를 다루는 도구.
from pathlib import Path
# 파이썬 실행 환경을 다룬다. 여기서는 import 검색 경로를 설정한다.
import sys

# 현재 main.py의 절대 경로를 구하고 그 파일이 들어 있는 폴더를 선택한다.
HERE = Path(__file__).resolve().parent
# 한 단계 위의 01_basic 폴더. 여기에 common 폴더가 있다.
LEVEL_ROOT = HERE.parent
# 모듈 검색 목록 맨 앞에 추가한다. str()은 경로를 문자열로 바꾼다.
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# 제곱근을 구하는 sqrt 함수를 사용하기 위해 가져온다.
import math
# 텐서(계산용 숫자 배열)를 만들고 행렬 계산을 하는 PyTorch 라이브러리.
import torch
# 그래프 라이브러리. 현재 기본 실습에는 그래프를 그리는 코드가 없어 사용하지 않는다.
import matplotlib
# 그래프 창 없이 이미지 파일로 저장할 때 쓰는 방식이다.
matplotlib.use("Agg")
# 그래프 도구를 plt라는 이름으로 가져온다. 이 파일에서는 직접 사용하지 않는다.
import matplotlib.pyplot as plt
# 파이썬 데이터를 JSON 파일로 저장하는 공통 함수.
from common.support import write_json

# 난수 시작값 고정. 다만 아래 계산은 정해진 숫자를 사용하므로 난수를 쓰지 않는다.
torch.manual_seed(42)
# PyTorch CPU 연산 스레드를 1개로 정한다.
torch.set_num_threads(1)

# main 함수를 정의한다. 맨 아래 main()에서 호출하면 내부 코드가 실행된다.
def main():
    # 토큰 3개에 각각 숫자 2개짜리 벡터를 준다. 크기(shape)는 [3, 2]이다.
    # 첫 토큰 [1, 0], 두 번째 [0, 1], 세 번째 [1, 1].
    # 1.처럼 소수점이 붙은 표기는 실수 1.0과 같다.
    q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    # clone()은 숫자가 같은 별도의 텐서를 만든다.
    # 오른쪽 결과 두 개를 각각 k와 v에 나눠 넣는다. 크기는 둘 다 [3, 2].
    k,v=q.clone(),q.clone()

    # TODO D20-05-B01: Scaled Dot-Product Q@K.T/sqrt(d_k)로 (3,3) Score를 계산하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-B01
    # @는 행렬 곱이다. 같은 위치의 숫자끼리 곱하는 연산(*)과 다르다.
    # k.T는 행과 열을 바꾸어 [3, 2]를 [2, 3]으로 만든다.
    # [3, 2] @ [2, 3] → [3, 3]: 모든 Query와 모든 Key의 내적을 계산한다.
    # 내적은 같은 위치 숫자끼리 곱하고 더한 값이다.
    # 예: 첫 Q [1, 0]과 세 번째 K [1, 1]의 내적은 1×1 + 0×1 = 1.
    # q.shape[-1]은 마지막 차원의 길이 2이다. sqrt(2)로 나눠 점수 규모를 조절한다.
    # scores의 행은 Query 위치, 열은 참조할 Key 위치이다.
    # 대략 [[0.707, 0, 0.707], [0, 0.707, 0.707], [0.707, 0.707, 1.414]].
    # 아직 확률이나 비율이 아니므로 행의 합이 1일 필요는 없다.
    scores = q @ k.T / math.sqrt(q.shape[-1]) # TODO: / math.sqrt(q.shape[-1])로 Scale 적용
    # TODO: assert scores.shape == (3, 3) 확인
    # assert는 조건이 거짓이면 AssertionError로 중단한다. 점수표 크기를 확인한다.
    assert scores.shape == (3, 3)
    # END D20-05-B01

    # TODO D20-05-B02: 미래 Key를 차단하는 upper-triangle Causal Mask를 작성하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-B02
    # 미래 토큰을 미리 보지 못하도록 막는 Causal Mask를 만든다.
    # ones(..., dtype=torch.bool)는 모든 칸이 True인 3×3 표이다.
    # triu(..., diagonal=1)는 대각선 바로 위부터 오른쪽 위 영역만 남긴다.
    # 결과: [[False, True,  True ],
    #        [False, False, True ],
    #        [False, False, False]]. 여기서는 True가 "차단"이라는 뜻이다.
    # 첫 토큰은 자신만, 두 번째는 첫째와 자신, 세 번째는 세 토큰 모두 참조한다.
    # zeros로 시작하면 모두 False라서 triu를 적용해도 아무것도 차단하지 못한다.
    mask = torch.triu(torch.ones((3, 3), dtype=torch.bool), diagonal=1)  # TODO: torch.triu(..., diagonal=1)
    # mask가 True인 점수만 음의 무한대(-inf)로 바꾼 새 텐서를 만든다.
    # Softmax에서 exp(-inf)=0이므로 해당 위치의 참조 비율이 0이 된다.
    # 점수를 0으로만 바꾸면 exp(0)=1이므로 완전히 차단되지 않는다.
    masked_scores = scores.masked_fill(mask,float("-inf")) # TODO: mask=True 위치를 -inf로 채우기
    # END D20-05-B02

    # TODO D20-05-B03: Score를 Softmax Weight로 바꾸고 Mask된 칸=0, 행 합=1을 확인하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-B03
    # Softmax는 각 점수의 지수값을 구하고 그 합으로 나누어 비율로 만든다.
    # dim=-1은 마지막 차원, 즉 여기서는 각 행의 열 방향으로 계산한다는 뜻이다.
    # 한 Query가 각 Key를 참고하는 비율이며, 한 행의 합은 1이다.
    # 대략 [[1, 0, 0], [0.330, 0.670, 0], [0.248, 0.248, 0.503]].
    # 각 행에 참조 가능한 위치가 적어도 하나 있어야 한다.
    # 한 행을 전부 -inf로 막으면 Softmax 결과에 NaN이 생길 수 있다.
    weights = torch.softmax(masked_scores, dim=-1)  # TODO: masked_scores 사용
    # 열 방향으로 더해 행마다 합을 구한다. [3, 3] → [3], 결과는 약 [1, 1, 1].
    row_sums = weights.sum(dim=-1)  # 각 Query 행의 합
    # TODO: assert torch.allclose(row_sums, torch.ones(3))
    # 실수 계산에는 작은 오차가 있어 == 대신 allclose로 충분히 가까운지 검사한다.
    # torch.ones(3)은 [1, 1, 1] 텐서이다.
    assert torch.allclose(row_sums, torch.ones(3))
    # TODO: assert torch.all(weights[mask] == 0)
    # weights[mask]는 True 위치의 값들만 골라낸다.
    # torch.all은 비교 결과가 모두 참인지 확인한다. 차단 위치의 비율은 전부 0이어야 한다.
    # mask가 전부 False면 선택값이 비어 이 검사만으로는 마스크 오류를 잡지 못한다.
    assert torch.all(weights[mask] == 0)
    # END D20-05-B03
    # 아래 값들은 후속 실습용 자리이다. 기본 단계에서는 추가 계산하지 않는다.
    # empty(0)은 원소가 0개인 텐서이다. tolist()로 바꾸면 []가 된다.
    plain_weights=torch.empty(0)
    # 마스크 유무에 따른 차이를 기록할 자리. None은 값이 없다는 뜻이다.
    diff=None
    # 여러 헤드의 결과를 기록할 자리. {}는 빈 딕셔너리이다.
    head_weights={}
    # Attention 비율만으로 최종 답변의 이유나 사실성을 모두 설명할 수는 없다.
    interpretation_limit=["Attention Weight만으로 모델의 최종 판단 전체를 설명할 수 없다"]

    # TODO D20-05-B04: Score·Causal Mask·Weight·행 합과 Output을 JSON으로 저장하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-B04
    # 정한 비율(weights)대로 실제 정보(v)를 섞는다.
    # [3, 3] @ [3, 2] → [3, 2]: 토큰마다 숫자 2개짜리 새 벡터가 나온다.
    # 예: 두 번째 출력 = 0.330×V첫째 + 0.670×V둘째 + 0×V셋째
    #                   ≈ 0.330×[1, 0] + 0.670×[0, 1] = [0.330, 0.670].
    # 이 출력은 다음 계산으로 전달할 정보 벡터이지, 생성한 단어나 정답은 아니다.
    output = weights @ v  # TODO: weights @ v
    # 저장할 결과를 딕셔너리로 모은다. 왼쪽 문자열은 항목 이름, 오른쪽은 값이다.
    # .tolist()는 텐서의 실제 숫자/불리언 값들을 파이썬 리스트로 바꾼다.
    # list(scores.shape)라면 실제 점수 대신 크기 [3, 3]만 얻게 된다.
    report = {
        # 차단 전 점수표. 위에서 계산한 scores 자체는 바뀌지 않았다.
        "scores": scores.tolist(),        # TODO: scores.tolist()
        # True가 차단 위치인 표.
        "causal_mask": mask.tolist(),   # TODO: mask.tolist()
        # 차단을 반영한 참조 비율.
        "weights": weights.tolist(),       # TODO: weights.tolist()
        # 각 행의 비율 합.
        "row_sums": row_sums.tolist(),      # TODO: row_sums.tolist()
        # 비율대로 V를 섞은 결과.
        "output": output.tolist(),        # TODO: output.tolist()
        # 후속 실습용 값들. 현재는 각각 [], None, {}이다.
        "no_mask_weights": plain_weights.tolist(),
        "masked_vs_unmasked_l1": diff,
        "multi_head": head_weights,
        "limits": interpretation_limit,
    }
    # Path의 /는 경로를 이어 붙인다. 현재 폴더의 outputs 아래에 저장한다.
    # write_json은 필요하면 폴더를 만들고 JSON 문자열로 변환해 UTF-8로 저장한다.
    # True/False/None은 JSON에서 true/false/null로 표현된다.
    write_json(HERE / "outputs" / "attention_report.json", report)
    # END D20-05-B04

# 이 파일을 직접 실행할 때만 main()을 호출한다.
# 다른 파일에서 import하면 함수 정의만 하고 main()은 자동 실행하지 않는다.
if __name__=="__main__":
    main()
