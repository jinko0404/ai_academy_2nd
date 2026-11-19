# 실습 목표 — 청크의 출처와 버전을 확인하고, 현재 검색에 공개된 청크 중 지정한 분류만 고릅니다.
# 전체 흐름 — 예제 3건 준비 → 필수 정보와 위치 확인 → 공개 버전 쌍 대조 → 분류 선택 → outputs/metadata_filter.json 저장.
"""실습 06 Intermediate · 버전 쌍과 분류 조건."""
from pathlib import Path
import sys
# 이 파일이 있는 폴더를 기준으로 공통 coursekit을 가져오고 보고서를 저장합니다.
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import save_json,show_success


# 메타데이터는 본문에 딸린 설명 정보입니다. 문서, 버전, 분류, 원문 위치를 다시 찾는 데 쓰입니다.
def require_metadata(row:dict)->None:
    # BEGIN D24-06-I01
    # TODO D24-06-I01 : 필수 Metadata 필드의 존재와 비어 있지 않음을 검사합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # row는 청크 한 건의 메타데이터입니다. 필수 키마다 문자열이 있는지 확인합니다.
    for key in ('doc_id','source_version','pipeline_version','source_locator','category','chunk_id'):
        # get()은 키가 없으면 None을 돌려줍니다. or의 첫 조건이 참이면 뒤의 strip()은 실행되지 않습니다.
        # strip()의 결과가 빈 문자열이면 입력이 비어 있거나 공백뿐이라는 뜻입니다.
        if not isinstance(row.get(key),str) or not row[key].strip():
            raise ValueError(f'missing {key}')
    # END D24-06-I01
# 문자 위치에는 순서가 있고 PDF 페이지는 1부터 셉니다. 페이지 개념이 없는 샘플에는 None이 들어갑니다.
def validate_locator(row:dict)->None:
    # BEGIN D24-06-I02
    # TODO D24-06-I02 : 문자 범위와 PDF page_number 등 자료형·조건을 확인합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 필수 문자열을 검사한 뒤, 원문에서 청크를 찾는 데 쓰는 문자 위치와 페이지를 확인합니다.
    require_metadata(row)
    # 시작 위치 a는 포함하고 끝 위치 b는 제외합니다. (0, 20)은 인덱스 0부터 19까지입니다.
    a,b=row.get('char_start'),row.get('char_end')
    if type(a) is not int or type(b) is not int or not 0<=a<b:
        raise ValueError('invalid span')
    # PDF처럼 페이지가 있는 문서는 1 이상의 정수를 사용합니다. 페이지가 없는 Markdown 예제는 None입니다.
    page=row.get('page_number')
    if page is not None and (type(page) is not int or page<1):
        raise ValueError('invalid page')
    # END D24-06-I02
# 원문 버전은 문서의 개정판이고 처리 버전은 청크를 만드는 규칙의 버전입니다. 두 값이 모두 맞는지 확인합니다.
def select_current(rows:list[dict],pointers:dict)->list[dict]:
    # BEGIN D24-06-I03
    # TODO D24-06-I03 : 현재 공개 중인 source_version·pipeline_version 쌍만 선택합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    chosen=[]
    for row in rows:
        # pointers는 문서 ID마다 현재 공개한 (원문 버전, 처리 버전)을 담은 딕셔너리입니다.
        # 문서 ID로 찾은 공개 버전 쌍과 현재 행의 두 값을 비교합니다.
        match=pointers.get(row['doc_id'])
        if match==(row['source_version'],row['pipeline_version']):
            # 튜플의 두 값이 모두 같으면 이 청크를 선택 목록에 넣습니다.
            chosen.append(row)
    return chosen
    # END D24-06-I03
# 현재 공개 버전에 해당하는 행에서 지정 분류를 고르고, 각 단계에서 빠진 이유를 기록합니다.
def filter_and_report(rows:list[dict],pointers:dict,category:str)->dict:
    # BEGIN D24-06-I04
    # TODO D24-06-I04 : category 필터 후 반환 건수와 제외 이유를 기록합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # valid에는 위치와 필수 필드가 올바른 행을, invalid에는 오류 난 청크 ID와 사유를 담습니다.
    valid=[];invalid=[]
    for row in rows:
        try:validate_locator(row)
        # 검사에서 나온 ValueError를 사유로 기록하고 다음 행도 계속 확인합니다.
        except ValueError as exc:invalid.append({'chunk_id':row.get('chunk_id'),'reason':str(exc)})
        else:valid.append(row)
    # 메타데이터 검사를 통과한 행을 대상으로 현재 공개 버전을 고릅니다.
    current=select_current(valid,pointers)
    # 공개 버전 필터를 통과한 행에서 category가 요청값과 같은 것만 담습니다.
    # category는 검색 분류 조건입니다. 실제 조회 권한은 별도의 인증 정보로 판단합니다.
    selected=[r for r in current if r['category']==category]
    # 유효 행에서 공개 행을 빼면 버전으로 제외된 건수가 됩니다.
    # 공개 행에서 선택 행을 빼면 분류로 제외된 건수가 됩니다.
    return {'total':len(rows),'selected':selected,'invalid':invalid,
            'excluded_other_version':len(valid)-len(current),
            'excluded_other_category':len(current)-len(selected)}
    # END D24-06-I04
def main():
    # 예제의 source_locator인 attendance.md#sec-1은 출석 안내 문서의 첫 번째 절을 가리킵니다.
    base={'doc_id':'DOC-ATTEND','source_locator':'attendance.md#sec-1','category':'학사',
          'pipeline_version':'ingest-e5-v1','char_start':0,'char_end':20,'page_number':None}
    # **base로 공통 키와 값을 복사하고, 각 예제의 버전이나 분류를 지정합니다.
    items=[{**base,'chunk_id':'C1','source_version':'v1'},
           {**base,'chunk_id':'C2','source_version':'v2'},
           {**base,'chunk_id':'C3','source_version':'v2','category':'시설'}]
    # 현재 공개 버전을 v2로 두고 학사를 선택합니다. C1은 버전에서, C3은 분류에서 빠져 C2만 남습니다.
    report=filter_and_report(items,{'DOC-ATTEND':('v2','ingest-e5-v1')},'학사')
    # 선택된 청크와 제외 사유별 건수를 한 JSON 파일에 저장합니다.
    out=save_json(HERE,'metadata_filter.json',report)
    show_success('06 Intermediate',out)
