# 실습 목적 — manifest(문서 목록)의 모든 파일을 확인하고 읽은 원본의 크기와 해시를 기록합니다.
# 전체 흐름 — 목록 읽기 → ID 검사와 정렬 → 확장자 확인 → 파일 읽기 → 상태별 집계 → JSON 저장.
"""실습 01 Intermediate · run via python main.py"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import DATA,save_json,show_success,load_json
from coursekit.documents import raw_path,sha_bytes


# rows의 각 항목은 문서 정보를 담은 딕셔너리입니다. 먼저 ID만 모아 중복을 확인합니다.
def sort_rows(rows:list[dict])->list[dict]:
    # BEGIN D24-01-I01
    # TODO D24-01-I01 : 문서 목록을 doc_id 순으로 정렬합니다. 동일 ID가 있으면 오류로 분류합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    ids = []
    for row in rows:
        ids.append(row["doc_id"])

    # set은 같은 값을 한 번만 보관하므로, 두 개수가 다르면 같은 ID가 두 번 이상 들어 있습니다.
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc ID")

    # sorted가 각 문서에서 어떤 값으로 순서를 정할지 알려 주는 함수입니다.
    def get_doc_id(row):
        return row["doc_id"]

    return sorted(rows, key=get_doc_id)
    # END D24-01-I01
def classify(path:Path)->str:
    # BEGIN D24-01-I02
    # TODO D24-01-I02 : 확장자와 허용 집합을 비교하여 supported/unsupported를 반환합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # suffix는 점을 포함한 확장자입니다. lower()로 .PDF도 .pdf와 같은 기준으로 비교합니다.
    extension = path.suffix.lower()
    supported_extension = {".txt", ".md", ".html", ".pdf"}

    if extension in supported_extension:
        return "supported"
    
    return "unsupported"
    # END D24-01-I02
def collect_one(doc:dict)->dict:
    # BEGIN D24-01-I03
    # TODO D24-01-I03 : 파일별 원문 크기·해시 또는 읽기 실패 이유를 안전하게 기록합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 원문 폴더에서 실제 경로를 찾고, 파일을 읽기 전에도 기록할 수 있는 정보를 먼저 채웁니다.
    # 입력의 source_version은 결과에서 version이라는 이름으로 보관합니다.
    path=raw_path(doc)
    result={'doc_id':doc['doc_id'],'file':doc['file'],'version':doc['source_version'],
            'category':doc['category'],'status':classify(path)}
    # 지원하지 않는 형식도 결과에 남깁니다. 여기서 반환하면 아래 파일 읽기 단계는 건너뜁니다.
    if result['status']=='unsupported':return result
    # read_bytes()는 파일의 원본 바이트를 그대로 읽습니다. 이 바이트로 크기와 해시를 계산합니다.
    try:raw=path.read_bytes()
    # 파일 없음이나 권한 문제로 읽기에 실패하면 FileNotFoundError 같은 예외 클래스 이름을 남깁니다.
    except OSError as exc:
        result.update(status='read_error',reason=type(exc).__name__)
    # try에서 읽기가 성공하면 실행됩니다. update()는 status를 바꾸고 크기와 해시 키를 추가합니다.
    else:
        result.update(status='collected',size_bytes=len(raw),raw_sha256=sha_bytes(raw))
    return result
    # END D24-01-I03
def collect_all(rows:list[dict])->dict:
    # BEGIN D24-01-I04
    # TODO D24-01-I04 : 모든 입력을 제외 없이 기록하고 상태별 건수 합이 입력 수와 일치하는지 검사합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # ID순으로 방문하면서 문서마다 collect_one의 결과를 한 건씩 새 목록에 담습니다.
    items=[collect_one(r) for r in sort_rows(rows)]
    # True를 1로 더하는 sum으로 상태별 건수를 구합니다. assert는 합계에서 빠진 상태가 있는지 검사합니다.
    counts={state:sum(r['status']==state for r in items)
            for state in ('collected','unsupported','read_error')}
    assert sum(counts.values())==len(items)
    # 전체 건수, 상태별 건수, 문서별 상세 내역을 한 딕셔너리로 반환합니다.
    return {'total':len(items),'counts':counts,'items':items}
    # END D24-01-I04
def main():
    result=collect_all(load_json(DATA/'manifest.json'))
    # save_json은 현재 실습 폴더의 outputs 아래에 결과 파일을 저장합니다.
    out=save_json(HERE,'manifest_report.json',result)
    show_success('01 Intermediate',out)
