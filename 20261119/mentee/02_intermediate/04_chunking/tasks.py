# 실습 목표 — 파싱된 본문을 일부 겹치는 청크로 나누고, 원문 출처와 E5 입력 형식을 붙입니다.
# 전체 흐름 — 문서 조회·파싱 → 단위별 정제·분할 → 청크에 출처 추가 → 선택적으로 토큰 측정·한도 검사 → outputs/chunking_report.json 저장.
# 청크를 나눌 때는 문자 수를 사용합니다. 모델의 입력 한도를 검사할 때는 토크나이저가 센 토큰 수를 사용합니다.
"""실습 04 Intermediate · --live-e5 시 실제 토크나이저 사용."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import save_json,show_success
from coursekit.documents import by_doc,parse_one
from coursekit.processing import clean_prose,char_windows,build_chunk,token_len


# 이 단계의 크기와 겹침은 문자 수입니다. 이후 검증을 위해 정제 본문과 그 위의 위치 목록을 함께 돌려줍니다.
def split_one(doc:dict,unit:dict,size=70,overlap=10)->tuple[str,list[dict]]:
    # BEGIN D24-04-I01
    # TODO D24-04-I01 : 하나의 ParsedUnit을 clean_text에서 문자 창으로 분할합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # unit은 파서가 꺼낸 본문 단위입니다. 청크의 문자 위치는 정제한 text를 기준으로 기록합니다.
    text = clean_prose(unit["raw_text"])
    # 기본 설정은 한 청크에 최대 70자, 인접 청크끼리 10자 겹침입니다.
    # char_windows가 범위를 하나씩 만들고, list()가 그 범위를 목록으로 모읍니다.
    return text, list(char_windows(text, size, overlap))
    # END D24-04-I01

# 문자 범위의 시작은 포함하고 끝은 제외합니다. E5에 넘길 입력은 그 범위의 본문에 접두어를 붙인 값입니다.
def attach_chunks(doc:dict,unit:dict,text:str,spans:list[dict])->list[dict]:
    # BEGIN D24-04-I02
    # TODO D24-04-I02 : 각 창에 doc_id·unit_id·버전·Locator와 passage: 접두어를 전달합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    chunks = []
    # span은 시작 위치, 끝 위치, 해당 범위의 본문을 담은 딕셔너리 한 건입니다.
    for span in spans:
        # build_chunk는 span의 위치와 본문을 검증하고 문서 ID, 버전, 출처와 E5 입력 문자열을 붙입니다.
        chunk = build_chunk(doc, unit, text, span)
        # 방금 만든 청크 딕셔너리 한 건을 목록에 추가합니다.
        chunks.append(chunk)

    return chunks

    # END D24-04-I02
# 토큰은 모델이 문자열을 읽는 단위입니다. 같은 문자열도 문자 수와 토큰 수가 달라질 수 있습니다.
def measure(chunks:list[dict],model=None)->list[dict]:
    # BEGIN D24-04-I03
    # TODO D24-04-I03 : SentenceTransformer 모델이 주어졌을 때 특수 토큰 포함 길이를 측정합니다. 모델이 없으면 not_run으로 기록합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    results = []

    if model is None:
        # 모델을 불러오지 않은 기본 실행에서도 각 청크에 토큰 측정 필드를 추가합니다.
        for chunk in chunks:
            # copy()로 기존 필드를 새 딕셔너리에 복사한 뒤 측정값을 기록합니다.
            row = chunk.copy()
            row["token_count"] = None
            # 실습에서 약속한 미측정 상태 문자열은 밑줄이 있는 not_run입니다.
            row["token_status"] = "not_run"
            results.append(row)
        return results

    for chunk in chunks:
        row = chunk.copy()
        # 모델에 넘길 embedding_text 전체를 셉니다. passage: 접두어와 토크나이저가 붙이는 특수 토큰도 포함됩니다.
        row["token_count"] = token_len(model, chunk["embedding_text"])
        row["token_status"] = "measured"
        results.append(row)
    
    return results

    # END D24-04-I03
# 측정한 토큰 수와 모델의 최대 입력 길이를 비교해 초과한 청크를 확인합니다.
def enforce_limit(chunks:list[dict],model=None)->dict:
    # BEGIN D24-04-I04
    # TODO D24-04-I04 : 최대 입력 길이를 초과한 청크를 통과시키지 않고 개수와 ID를 기록합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    if model is None:
        return{
            "status": "not_run",
            "too_long_ids": [],
            "checked": 0
        }

    # 토큰 수가 모델 한도를 넘은 청크의 ID만 모아 재분할할 대상을 찾습니다.
    too_long_ids = []

    for chunk in chunks:
        token_count = chunk["token_count"]

        # 측정값이 없는 청크는 continue로 넘어갑니다.
        if token_count is None:
            continue

        # 토큰 수가 최대 길이와 같으면 허용됩니다. 최대 길이를 초과할 때만 기록합니다.
        if token_count > model.max_seq_length:
            too_long_ids.append(chunk["chunk_id"])

    # 초과한 ID가 하나라도 있으면 failed, 초과 목록이 비어 있으면 pass로 기록합니다.
    if too_long_ids:
        status = "failed"
    else:
        status = "pass"
    
    return{
        "status": status,
        "too_long_ids": too_long_ids,
        # checked에는 입력 청크 전체 수를 기록합니다. token_count가 None이라 비교를 건너뛴 행도 포함됩니다.
        "checked": len(chunks)
    }
    # END D24-04-I04
def main():
    # 기본 실행은 준비한 청크를 확인합니다. --live-e5를 넘기면 실제 모델의 토크나이저로 길이도 측정합니다.
    model=None
    if '--live-e5' in sys.argv:
        from coursekit.live import load_encoder
        model=load_encoder()
    all_chunks=[]
    # 문서마다 추출 단위를 따로 처리합니다. 예를 들어 출석 안내의 절과 과제 안내의 본문은 각각 분할합니다.
    for doc_id in ('DOC-ATTEND','DOC-ASSIGN'):
        doc=by_doc(doc_id)
        for unit in parse_one(doc)['units']:
            text,spans=split_one(doc,unit)
            # extend는 한 단위에서 만든 청크를 하나씩 기존 목록에 추가합니다.
            all_chunks.extend(attach_chunks(doc,unit,text,spans))
    measured=measure(all_chunks,model)
    report=enforce_limit(measured,model)
    # 보고서에는 측정 필드가 붙은 전체 청크 목록과 한도 검사 요약을 함께 저장합니다.
    out=save_json(HERE,'chunking_report.json',{'chunks':measured,'token_validation':report})
    show_success('04 Intermediate',out)
