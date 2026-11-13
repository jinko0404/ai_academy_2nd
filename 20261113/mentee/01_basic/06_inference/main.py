
# 이 프로그램은 후보 세 개 중에서 하나를 고르는 연습이다.
# 후보 이름은 출력A, 출력B, 출력C이고 점수는 사람이 직접 2점, 1점, 0점으로 정했다.
# 실제 AI 모델을 불러오거나 질문에 답하는 프로그램은 아니다.
#
# 전체 흐름: 아래 다섯 단계로 읽으면 된다.
#
# 1. 후보별 점수 준비: A=2점, B=1점, C=0점
#                  ↓
# 2. 점수를 확률로 바꿈: A 약 66.5%, B 약 24.5%, C 약 9.0%
#                  ↓
# 3. 온도를 바꿨을 때 확률이 어떻게 달라지는지도 별도로 계산
#                  ↓
# 4. 2번의 원래 확률에서 가장 큰 후보 선택 → 출력A
#                  ↓
# 5. 점수, 확률, 선택한 이름을 inference_report.json 파일에 저장
#
# 3번은 비교용이다. 4번에서는 3번의 결과가 아니라 2번의 확률을 쓴다.
# 확률을 계산한다고 무작위로 뽑는 것은 아니다. 여기서는 가장 큰 값만 고른다.
#
# 아래는 실행 준비이다. 후보를 고르는 계산은 def main():부터 읽으면 된다.
# Path는 파일을 저장할 위치를 다루는 도구이다.
from pathlib import Path
# sys는 파이썬 실행 환경을 설정하는 도구이다.
import sys

# __file__은 이 파일의 위치, resolve()는 전체 경로, parent는 파일이 들어 있는 폴더.
# 따라서 HERE는 06_inference 폴더이다.
HERE = Path(__file__).resolve().parent
# 한 단계 위인 01_basic 폴더. common 폴더가 여기에 있다.
LEVEL_ROOT = HERE.parent
# 파이썬에게 이 폴더에서도 코드를 찾아보라고 알려준다.
# str()은 경로를 문자열로 바꾸고 insert(0, ...)은 목록 맨 앞에 넣는다.
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# torch는 여러 숫자를 한꺼번에 계산하는 도구이다.
import torch
# F라는 짧은 이름으로 도구를 가져온다. 이 파일에서는 F를 사용하지 않는다.
import torch.nn.functional as F
# 계산 결과를 JSON 파일로 저장해 주는 함수를 가져온다.
from common.support import write_json

# 무작위 계산의 시작값 설정. 아래 계산에는 무작위 추첨이 없어 결과에 영향 없다.
torch.manual_seed(42)
# CPU 계산에 사용할 스레드를 1개로 정한다. 후보 선택 규칙과는 관계없다.
torch.set_num_threads(1)

def main():
    # [1단계] 후보별 점수와 이름을 준비한다.
    # logits는 "확률로 바꾸기 전 점수"라고 생각하면 된다.
    # 2.0은 2점이지 200%가 아니다.
    # torch.tensor(...)는 숫자 목록을 torch가 계산할 수 있는 형태인 텐서로 만든다.
    logits=torch.tensor([2.0,1.0,0.0])
    # labels는 이름을 순서대로 담은 리스트이다. 점수와 순서가 대응한다.
    # 파이썬은 위치를 0부터 센다.
    # 0번: 출력A ↔ 2점 / 1번: 출력B ↔ 1점 / 2번: 출력C ↔ 0점.
    labels=["출력A","출력B","출력C"]

    # TODO D20-06-B01: Logit[2,1,0]을 Softmax로 바꾸어 확률의 합이 1인지 확인하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B01
    # [2단계] 점수 세 개를 확률 세 개로 바꾸고 prob에 저장한다.
    # softmax는 큰 점수에 큰 확률을 주면서 전체 합을 1로 맞추는 계산이다.
    # 단순히 각 점수를 총점으로 나누는 계산은 아니다.
    # 결과는 약 [0.6652, 0.2447, 0.0900]. 순서는 여전히 A, B, C이다.
    # 0.6652는 약 66.52%라는 뜻이다.
    # dim=0은 이 한 줄짜리 숫자 목록을 대상으로 계산하라는 뜻이다.
    prob = torch.softmax(logits, dim=0)  # TODO: torch.softmax(logits, dim=0)
    # TODO: assert torch.isclose(prob.sum(), torch.tensor(1.0), atol=1e-6)
    # prob.sum()은 확률 세 개를 더한다. 결과는 약 1이어야 한다.
    # torch.tensor(1.0)은 비교할 기준 1을 torch 형태로 만든다.
    # isclose는 두 값이 충분히 가까운지 확인한다.
    # 소수 계산에는 작은 오차가 있어 정확히 == 1로 비교하지 않는다.
    # atol=1e-6은 절대 오차 허용값 0.000001이다. 기본 상대 오차 허용값도 적용된다.
    # assert는 검사 결과가 참이면 넘어가고, 거짓이면 오류를 내고 멈춘다.
    assert torch.isclose(prob.sum(), torch.tensor(1.0), atol=1e-6)
    # END D20-06-B01

    # TODO D20-06-B02: Temperature 0.5·1.0·2.0에서 확률 분포를 비교하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B02
    # [3단계] 온도를 달리하면 확률이 어떻게 변하는지 비교한다.
    # 온도는 실제 온도가 아니라 점수를 나눌 숫자이다.
    # 작은 온도는 점수 차이를 키우고, 큰 온도는 점수 차이를 줄인다.
    # {}는 빈 딕셔너리이다. 이름을 붙여 값을 보관하는 자료형이다.
    # 예: {"0.5": [확률 세 개], "1.0": [확률 세 개], ...}를 만들 것이다.
    temperature_results = {}
    # 아래 계산을 세 번 한다. temperature에 0.5, 1.0, 2.0이 차례로 들어간다.
    for temperature in (0.5, 1.0, 2.0):
        # 길어 보이지만 오른쪽 계산 결과를 왼쪽 보관함에 넣는 한 문장이다.
        # 첫 반복인 temperature=0.5를 예로 들어 보자.
        # ① logits / temperature → [2, 1, 0] / 0.5 → [4, 2, 0].
        #    텐서를 숫자로 나누면 안의 점수를 각각 나눈다.
        # ② softmax(...) → 약 [0.8668, 0.1173, 0.0159].
        # ③ .tolist() → 확률 텐서를 평범한 파이썬 리스트로 바꾼다.
        # ④ str(temperature) → 숫자 0.5를 문자열 "0.5"로 바꾼다.
        # ⑤ temperature_results["0.5"] = 위에서 구한 확률 리스트.
        # dim=-1은 마지막 방향이다. 지금은 한 줄 목록이라 dim=0과 같다.
        # 괄호 안에서는 여러 줄로 적어도 하나의 계산이다.
        temperature_results[str(temperature)] = torch.softmax(
            logits / temperature, dim=-1
        ).tolist()  # TODO: softmax(logits/temperature) 결과
    # 세 번 반복한 뒤의 내용은 대략 다음과 같다. 확률 순서는 A, B, C이다.
    # {"0.5": [0.8668, 0.1173, 0.0159],
    #  "1.0": [0.6652, 0.2447, 0.0900],
    #  "2.0": [0.5065, 0.3072, 0.1863]}
    # 온도를 바꿔도 A > B > C 순서는 같고 비율만 달라진다.
    # 결과를 따로 보관했으므로 위에서 만든 prob는 바뀌지 않는다.
    # {"0.5": [0.8668, 0.1173, 0.0159],
    #  "1.0": [0.6652, 0.2447, 0.0900],
    #  "2.0": [0.5065, 0.3072, 0.1863]}
    # END D20-06-B02

    # TODO D20-06-B03: Greedy로 다음 Token 후보를 선택하고 학습/추론 차이를 기록하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B03
    # [4단계] 원래 확률 prob에서 가장 큰 후보의 위치를 찾는다.
    # Greedy는 "그때 가장 높은 확률의 후보를 고르기"라는 방법 이름이다.
    # prob는 약 [0.6652, 0.2447, 0.0900]이고 가장 큰 값은 첫 번째이다.
    # argmax(prob) → tensor(0). 큰 값 자체가 아니라 그 값의 위치 0을 구한다.
    # .item() → tensor(0)에서 파이썬 숫자 0을 꺼낸다.
    # int(...) → 정수로 만든다. 여기서는 이미 정수라 그대로 0이다.
    # 따라서 selected_index에는 정수 0이 들어간다.
    selected_index = int(torch.argmax(prob).item())   # TODO: int(torch.argmax(prob).item())
    # labels[selected_index]는 labels[0]과 같다. 첫 번째 이름을 꺼낸다.
    # 따라서 selected_token에는 문자열 "출력A"가 들어간다.
    # 토큰은 모델이 처리하는 글자/단어 조각 등을 뜻한다. 여기서는 예시 이름을 썼다.
    selected_token = labels[selected_index]     # TODO: labels[selected_index]
    # 아래는 실행할 명령이 아니라 저장할 설명 문장들이다.
    # 학습: 틀린 정도를 구하고 모델 내부의 숫자 설정을 고쳐 간다.
    # loss=틀린 정도, backward=고칠 방향 계산, 업데이트=실제 숫자 변경.
    # 추론: 배운 숫자 설정(가중치)으로 출력을 구하며 그 설정을 고치지 않는다.
    # 자동회귀: 방금 만든 토큰을 입력 뒤에 붙이고 다음 토큰을 또 구하는 방식이다.
    # 이 예제는 후보 한 번 선택까지만 한다. 학습이나 반복 생성은 실행하지 않는다.
    training_vs_inference = {
        "training": "loss·backward·파라미터 업데이트",       # TODO: loss·backward·파라미터 업데이트
        "inference": "가중치를 바꾸지 않고 다음 Token 예측",      # TODO: 가중치를 바꾸지 않고 다음 Token 예측
        "autoregressive": "생성 Token을 다시 문맥에 추가", # TODO: 생성 Token을 다시 문맥에 추가
    }
    # END D20-06-B03
    # 아래는 후속 실습용 빈 보관함이다. 계산을 빼먹은 것이 아니다.
    # top_k는 높은 점수 몇 개만 후보로 남기는 방법이다. 여기서는 실행하지 않는다.
    # {}는 빈 딕셔너리이다.
    top_k={}
    # sampling은 확률에 따라 무작위로 뽑는 방법이다. 여기서는 실행하지 않는다.
    # None은 값이 없음을 나타낸다. 선택 위치와 이름을 모두 비워 둔다.
    sampled_index=None
    sampled_token=None
    # top_p는 확률이 높은 후보부터 합을 모아 기준에 도달할 때까지 남기는 방법이다.
    # 여기서는 실행하지 않아 {}이다.
    top_p={}
    # 실제 학습 결과를 담을 자리이다. 학습하지 않으므로 {}이다.
    training_check={}
    # KV 캐시는 토큰을 이어서 만들 때 이전 계산 정보를 재사용하는 장치이다.
    # 아래에는 설명만 적었다. 실제 캐시를 만들거나 사용하는 코드는 없다.
    kv_note={"stores":"이전 K/V 재사용","not":"학습이나 장기 기억이 아님"}

    # TODO D20-06-B04: 확률·선택·학습/추론 차이를 inference_report.json에 기록하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-B04
    # [5단계] 결과를 report라는 딕셔너리 하나로 모은다.
    # "항목 이름": 값 형태이다. 왼쪽은 이름, 오른쪽은 보관할 값이다.
    report = {
        # 원래 점수 텐서를 [2.0, 1.0, 0.0]이라는 리스트로 바꿔 넣는다.
        "logits": logits.tolist(),
        # 원래 점수로 계산한 확률 세 개를 리스트로 바꿔 넣는다.
        "probs": prob.tolist(),          # TODO: prob.tolist()
        # 온도 세 가지의 확률 계산 결과 전체를 넣는다.
        "temperature": temperature_results,
        # 안쪽에도 딕셔너리를 넣을 수 있다. 여기서는 {"index": 0, "token": "출력A"}.
        "greedy": {"index": selected_index, "token": selected_token},
        # 무작위 선택을 하지 않아 index와 token은 둘 다 None이다.
        "sampling": {"index": sampled_index, "token": sampled_token},
        # 실행하지 않은 두 방법의 빈 보관함을 넣는다.
        "top_k": top_k,
        "top_p": top_p,
        # 학습과 추론에 관한 설명 문장들을 넣는다.
        "training_vs_inference": training_vs_inference,
        # 실제 학습 결과가 없어 {}가 들어간다.
        "training_step": training_check,
        # 실제 캐시 데이터가 아니라 설명 문장들이 들어간다.
        "kv_cache": kv_note,
    }
    # HERE / "outputs" / "inference_report.json"은 저장할 파일 위치이다.
    # 여기서 /는 나눗셈이 아니라 폴더와 파일 경로를 이어 붙이는 연산이다.
    # 이 main.py와 같은 폴더 아래 outputs/inference_report.json으로 저장한다.
    # write_json이 필요하면 폴더를 만들고 report를 JSON 형식의 글로 쓴다.
    # JSON은 딕셔너리처럼 항목 이름과 값을 적는 파일 형식이다.
    # 파이썬의 None은 JSON에서 null로 적힌다.
    # print가 없어 정상 실행해도 완료 문장이 화면에 뜨지 않는다. 파일을 확인하자.
    write_json(HERE / "outputs" / "inference_report.json", report)
    # END D20-06-B04

    # 여기까지 실행되면 결과 파일을 저장한 뒤 함수가 끝난다.

# 이 파일을 직접 실행할 때만 아래 main()을 호출한다.
# 다른 파일에서 import해서 가져오면 main()을 자동 실행하지 않는다.
if __name__=="__main__":
    # 여기서 위의 main 함수로 들어가 1단계부터 차례대로 실행한다.
    main()
