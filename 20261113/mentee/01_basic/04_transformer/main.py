# 전체 흐름도
#
# token_ids [2, 5]: 입력 2개, 입력당 토큰 위치 5개
#     │
#     ├── ID == 0 비교 ──→ pad_mask [2, 5]: PAD 위치만 True
#     │                         │
#     ▼                         │
# Embedding: ID 하나를 숫자 8개짜리 벡터로 변환
#     │                         │
#     ▼                         │
# x [2, 5, 8]                   │
#     │                         │
#     ▼                         ▼
# TransformerEncoderLayer(x, src_key_padding_mask=pad_mask)
#     │ Self-Attention: 같은 입력 안의 토큰 사이에서 정보를 섞음
#     │ FFN: 각 위치의 특징을 변환 (8 → 16 → 8)
#     │ 잔차 연결과 정규화도 레이어 내부에서 수행
#     ▼
# y [2, 5, 8] → x와 y의 크기가 같은지 검사
#     │
#     ▼
# 크기 정보와 마스크를 리스트로 변환 → report 딕셔너리 → JSON 저장
#
# B=Batch(입력 묶음 수), T=토큰 위치 수, D=벡터당 숫자 개수.
# 크기가 같아도 x와 y의 실제 숫자 값은 달라질 수 있다.
# 무작위로 초기화한 Encoder 한 층의 실습이며 학습이나 문장 생성은 수행하지 않는다.
# 위치 임베딩도 이 코드에서는 추가하지 않는다.

# Path는 파일과 폴더의 경로를 다루는 도구이다.
from pathlib import Path
# sys는 실행 환경을 다룬다. 여기서는 import 검색 경로를 설정한다.
import sys

# __file__은 현재 파일 경로, resolve()는 절대 경로, parent는 상위 폴더이다.
HERE = Path(__file__).resolve().parent
# common 폴더가 들어 있는 01_basic 폴더를 가리킨다.
LEVEL_ROOT = HERE.parent
# 검색 경로 목록 맨 앞에 추가한다. str()은 경로를 문자열로 바꾼다.
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# 계산용 숫자 배열인 텐서를 다루는 PyTorch 라이브러리.
import torch
# nn에는 Embedding, TransformerEncoderLayer 등의 신경망 구성요소가 있다.
from torch import nn
# 파이썬 데이터를 JSON 파일로 저장하는 공통 함수.
from common.support import write_json

# 무작위 초기화를 재현하기 위해 난수 시작값을 정한다.
# 같은 환경과 실행 순서에서 결과를 비교하기 쉽게 만든다.
torch.manual_seed(42)
# PyTorch CPU 연산에 사용할 스레드 수를 1로 설정한다.
torch.set_num_threads(1)

# 함수를 정의한다. 맨 아래 main()에서 호출하면 들여쓴 코드들이 실행된다.
def main():
    # 바깥 리스트에는 입력 2개, 각 안쪽 리스트에는 토큰 ID 5개가 들어 있다.
    # 이 실습에서는 2=BOS, 3=EOS, 0=PAD이며, 4~7은 예시 토큰 ID이다.
    # ID는 토큰을 구분하는 번호이다. 큰 번호라고 더 중요한 토큰은 아니다.
    # torch.tensor는 리스트를 텐서로 바꾸고, torch.long은 64비트 정수 자료형이다.
    token_ids=torch.tensor([[2,4,5,3,0],[2,6,7,3,0]],dtype=torch.long)

    # TODO D20-04-B01: Token ID 텐서와 nn.Embedding 출력의 shape (B,T,D)를 구하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-B01
    # ID 0~15에 대응하는 벡터 16개를 보관하는 표를 만든다. 표 크기는 [16, 8].
    # 각 ID에 숫자 8개짜리 벡터가 대응한다. 같은 ID는 같은 표의 행을 가져온다.
    # 표는 무작위로 초기화되며 여기서는 학습으로 갱신하지 않는다.
    # padding_idx를 지정하지 않았으므로 ID 0의 벡터도 자동으로 0이 되지 않는다.
    embedding = nn.Embedding(num_embeddings=16, embedding_dim=8)  # ID 16종, 특성 8개
    # ID마다 벡터를 꺼낸다. [2, 5]의 각 위치에 숫자 8개가 붙어 [2, 5, 8]이 된다.
    x = embedding(token_ids)       # TODO: embedding(token_ids) 호출
    # .shape는 실제 값이 아니라 크기 정보이다. list(...)로 리스트로 바꾼다.
    # 여기서는 [2, 5]가 된다.
    id_shape = list(token_ids.shape)  # TODO: token_ids.shape을 리스트로 변환
    # x의 크기를 리스트로 바꾸면 [2, 5, 8]이다.
    # x 안의 숫자 값들을 리스트로 바꾸고 싶다면 x.tolist()를 사용한다.
    emb_shape = list(x.shape) # TODO: x.shape을 리스트로 변환
    # END D20-04-B01

    # TODO D20-04-B02: EncoderLayer(d_model=8,nhead=2)로 전달하고 shape 보존을 검증하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-B02
    # Encoder 한 층을 만든다. 생성만으로 x를 처리하는 것은 아니다.
    # d_model=8: 입력/출력 벡터 길이. Embedding의 embedding_dim과 같아야 한다.
    # nhead=2: Attention을 두 헤드로 나눠 계산한다. 헤드당 차원은 8 / 2 = 4.
    # d_model은 nhead로 나누어떨어져야 한다.
    # dim_feedforward=16: FFN 내부에서 8차원을 16차원으로 늘렸다가 8로 되돌린다.
    # batch_first=True: 입력을 [B, T, D] 순서로 받는다.
    # dropout=0.0: 이 레이어의 Dropout 비율을 0으로 정한다.
    layer = nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, batch_first=True, dropout=0.0
    )
    # 평가 모드로 전환한다. eval() 자체가 기울기 계산 기록을 끄는 것은 아니다.
    layer.eval()
    # == 비교를 모든 위치에 적용한다. ID가 0이면 True, 아니면 False이다.
    # 결과: [[False, False, False, False, True],
    #        [False, False, False, False, True]]. 크기는 [2, 5]이다.
    # 이 마스크는 True 위치를 Attention의 참조 대상(key)에서 제외한다.
    # 앞 실습의 attention_mask(유효=1, PAD=0)와 표시 방식이 다르다.
    pad_mask = token_ids == 0  # PAD 위치는 True로 표시
    # with 블록 안에서는 기울기 계산을 위한 기록을 남기지 않는다.
    # no_grad(layer(...)) 대신 아래 형태로 사용한다. y는 레이어의 출력 텐서이다.
    # PAD 위치를 삭제하지 않으므로 출력에도 입력당 5개 위치가 남는다.
    # PAD 위치의 출력값이 전부 0이라는 보장도 없다.
    # 미래 토큰 참조를 막는 Causal Mask는 이 코드에서 전달하지 않는다.
    with torch.no_grad():
        y = layer(x, src_key_padding_mask=pad_mask)
    # 출력의 크기는 입력 x와 같은 [2, 5, 8]이다.
    out_shape = list(y.shape)  # TODO: y.shape을 리스트로 변환
    # TODO: assert로 Embedding 입력/Encoder 출력 shape가 같은지 비교
    # assert는 조건이 거짓이면 AssertionError를 낸다.
    # 크기만 비교한다. 벡터 안의 숫자 값이 같은지는 검사하지 않는다.
    assert x.shape == y.shape
    # END D20-04-B02
    # 레이어의 역할과 출력 크기를 설명용 딕셔너리로 묶는다.
    components={"attention":"토큰 간 정보","ffn":"특징 변환","shape":out_shape}
    # 아래 빈 값들은 후속 실습용 자리이다. 기본 단계에서는 추가 계산하지 않는다.
    # []는 빈 리스트, {}는 빈 딕셔너리, None은 값이 없음을 나타낸다.
    # PAD 위치들을 모아 둘 자리.
    padding_positions=[]
    # PAD 위치의 출력 절댓값 합을 기록할 자리. 여기서는 계산하지 않는다.
    pad_values=None
    # PAD Mask는 채우기용 위치, Causal Mask는 미래 위치를 참조하지 못하게 한다.
    padding_note="PAD Mask는 Causal Mask와 별개"
    # 차원과 헤드 수가 맞지 않을 때의 오류 실험 결과를 기록할 자리.
    head_divisibility_error=None
    # 다른 입력 크기로 실험한 결과를 기록할 자리.
    extra_shapes={}
    # 코드가 수행하는 범위를 설명한다. 완성된 Decoder LLM은 아니다.
    architecture_note="이 코드는 Encoder Layer의 Shape 실습"

    # TODO D20-04-B03: 모든 shape와 Mask 의미를 transformer_report.json에 정리하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-B03
    # 저장할 결과를 딕셔너리 하나로 모은다.
    # "ids": id_shape에서 왼쪽은 저장 항목 이름, 오른쪽은 넣을 값이다.
    report = {
        # 아래 세 항목은 실제 ID/벡터 값이 아니라 크기 정보이다.
        "ids": id_shape,             # TODO: id_shape
        "embedding": emb_shape,       # TODO: emb_shape
        "encoder_output": out_shape,  # TODO: out_shape
        # .tolist()는 텐서 안의 True/False 값들을 중첩 리스트로 바꾼다.
        # list(pad_mask.shape)라면 실제 마스크 대신 크기 [2, 5]만 얻게 된다.
        "pad_key_mask": pad_mask.tolist(),    # TODO: pad_mask.tolist()
        # 아래 항목들은 앞서 만든 설명과 후속 실습용 값들을 그대로 담는다.
        "components": components,
        "padding_positions": padding_positions,
        "padding_output_abs_sum": pad_values,
        "padding_note": padding_note,
        "bad_head_rejected": head_divisibility_error,
        "extra_shapes": extra_shapes,
        "architecture_note": architecture_note,
    }
    # Path의 /는 경로를 이어 붙인다. 현재 폴더의 outputs 아래에 저장한다.
    # write_json이 필요하면 폴더를 만들고 JSON 문자열로 변환해 UTF-8로 저장한다.
    # True/False/None은 JSON에서 true/false/null로 표현된다.
    write_json(HERE / "outputs" / "transformer_report.json", report)
    # END D20-04-B03

# 이 파일을 직접 실행할 때만 main()을 호출한다.
# 다른 파일에서 import할 때는 main()을 자동 실행하지 않는다.
if __name__=="__main__":
    main()
