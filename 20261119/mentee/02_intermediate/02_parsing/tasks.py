# 실습 목표 — TXT·Markdown·HTML·PDF에서 읽을 수 있는 본문을 꺼내고, 파일 안에서 그 내용을 다시 찾을 수 있게 기록합니다.
# ParsedUnit은 추출한 텍스트와 출처를 묶은 기록입니다. 이 실습에서는 TXT 파일 전체, Markdown 절, HTML 본문, PDF 페이지가 각각 추출 단위가 됩니다.
# 전체 흐름 — main.py 실행 → manifest 읽기 → 문서별 형식 확인 → 공통 파서로 본문·출처 추출 → 상태 집계 → outputs/parsing_report.json 저장.
"""실습 02 Intermediate · format dispatch와 공통 ParsedUnit."""
from pathlib import Path
import sys
# 이 파일의 폴더를 기준으로 수강생 루트 경로를 찾아 등록합니다. 그래야 coursekit을 import할 수 있습니다.
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import DATA,load_json,save_json,show_success
from coursekit.documents import parse_one,raw_path


# 파일 확장자를 기준으로 어느 처리 함수에 문서를 맡길지 결정하는 단계입니다.
def kind_of(doc:dict)->str:
    # BEGIN D24-02-I01
    # TODO D24-02-I01 : 파일 확장자로 파서 경로를 결정합니다. 미지원 확장자를 정상 취급하지 않습니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 예를 들어 attendance.MD에서 suffix는 .MD이고, lower()를 적용하면 .md가 됩니다.
    suffix=Path(doc['file']).suffix.lower()

    # 딕셔너리의 키는 실제 확장자이고, 값은 아래 함수들이 공통으로 사용하는 형식 이름입니다.
    # get(suffix, 기본값)은 확장자가 표에 없을 때 unsupported를 돌려줍니다.
    return {
        '.txt':'txt',
        '.md':'markdown',
        '.html':'html',
        '.pdf':'pdf'
     }.get(suffix,'unsupported')
    # END D24-02-I01

# 정상 텍스트, 빈 텍스트, 읽기나 디코딩에 실패한 경우를 각각 결과에서 확인할 수 있어야 합니다.
def parse_txt(doc:dict)->dict:
    # BEGIN D24-02-I02
    # TODO D24-02-I02 : TXT를 UTF-8로 읽고 오류와 빈 결과를 구분합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # TXT 전용 함수에 다른 종류의 문서를 넘기면 ValueError로 잘못된 호출을 알립니다.
    if kind_of(doc) != 'txt':
        raise ValueError("wrong parser")

    # 공통 파서가 TXT를 UTF-8 계열로 읽어 한 개의 ParsedUnit으로 만듭니다.
    # 본문이 공백뿐이면 empty_text_review, 파일을 읽거나 디코딩하는 데 실패하면 해당 오류 상태가 돌아옵니다.
    return parse_one(doc)

    # END D24-02-I02
# Markdown은 제목이 나누는 절을, HTML은 주요 본문과 표의 열 이름 및 값을 살피세요.
# source_locator는 추출한 내용의 원문 위치를 다시 찾는 데 사용합니다.
def parse_markup(doc:dict)->dict:
    # BEGIN D24-02-I03
    # TODO D24-02-I03 : Markdown·HTML에서 제목·본문·표의 구조를 가진 ParsedUnit을 추출합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 여기서는 kind_of()가 반환하는 형식 이름을 비교합니다. .md는 markdown이라는 이름으로 전달됩니다.
    if kind_of(doc) not in ('markdown', 'html'):
        raise ValueError("wrong parser")
    
    # 공통 파서는 Markdown의 절 제목과 본문을 짝지어 남깁니다. HTML은 main의 내용을 읽고 표의 열 이름과 값을 함께 남깁니다.
    return parse_one(doc)
    # END D24-02-I03
# dispatch는 종류에 맞는 처리 함수로 보내는 역할입니다. PDF의 페이지 위치와 OCR 필요 상태도 이 단계에서 다룹니다.
def parse_dispatch(doc:dict)->dict:
    # BEGIN D24-02-I04
    # TODO D24-02-I04 : PDF는 페이지별 위치를 기록하고 이미지 PDF는 별도 needs_ocr로 분류합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 문서 종류를 한 번 구한 뒤, 조건이 맞는 곳에서 해당 함수의 결과를 바로 반환합니다.
    kind=kind_of(doc)
    if kind=='txt':
        return parse_txt(doc)
    if kind in {'markdown','html'}:
        return parse_markup(doc)
    # 공통 파서가 PDF의 페이지 번호와 파일명#page=1 같은 위치를 기록합니다.
    # 텍스트층이 없는 수업용 scan.pdf는 needs_ocr 상태로 기록됩니다.
    if kind=='pdf':
        return parse_one(doc)
    
    # 지원하는 종류에 해당하지 않으면 문서 ID와 실패 이유를 보고서에 남기고 units는 빈 목록으로 둡니다.
    return {'doc':{'doc_id':doc['doc_id']},'status':'unsupported','units':[],
            'reason':'unsupported_extension'}
    # END D24-02-I04
def main():
    # manifest.json에는 문서 ID, 파일명, 분류, 원문 버전이 담겨 있습니다.
    rows=load_json(DATA/'manifest.json')
    # 목록 표현식은 문서마다 parse_dispatch를 한 번씩 호출해 그 결과를 차례대로 담습니다.
    # 결과 하나는 doc(문서 정보), status(처리 상태), units(추출 단위), reason(사유)을 담습니다.
    results=[parse_dispatch(r) for r in rows]
    # 중괄호 안에서 먼저 실제로 등장한 상태를 모은 뒤, 상태마다 일치하는 결과의 수를 셉니다.
    # 출력의 items는 파일별 상세 내역이고 status_counts는 상태별 건수입니다.
    # 제공 샘플의 기대 건수는 parsed 4건, needs_ocr 1건, unsupported 1건입니다.
    out=save_json(HERE,'parsing_report.json',{'items':results,
             'status_counts':{s:sum(r['status']==s for r in results)
                for s in sorted({r['status'] for r in results})}})
    show_success('02 Intermediate',out)
