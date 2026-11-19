# 실습 목표 — 일반 본문을 정제하면서 숫자·조건·문단이 유지되는지 확인합니다.
# 전체 흐름 — 출석 안내 파싱 → 첫 번째 추출 단위 정제 → 핵심 구절 확인 → clean_report.json 저장.
"""실습 03 Intermediate · 정제 계약과 보고서."""
from pathlib import Path
import sys, re,unicodedata,hashlib
# 이 파일의 폴더를 기준으로 공통 coursekit을 불러오고, 나중에 이 실습 폴더에 결과를 저장합니다.
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import DATA,save_json,show_success
from coursekit.documents import by_doc,parse_one


# NFC는 ᄀ과 ᅡ처럼 같은 글자를 여러 코드 포인트로 표현한 입력을 가처럼 통일합니다.
def apply_nfc(text:str)->str:
    # BEGIN D24-03-I01
    # TODO D24-03-I01 : 원문을 Unicode NFC로 정규화합니다. NFKC로 동그라미 번호를 무조건 지우지 않습니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # unicodedata는 여러 함수를 담은 모듈입니다. NFC는 모듈 안의 unicodedata.normalize('NFC', text)로 적용합니다.
    # 첫 인자는 정규화 방식이고, 두 번째 인자는 바꿀 문자열입니다.
    return unicodedata.normalize('NFC', text)
    # END D24-03-I01
# Windows의 CRLF와 CR을 LF로 통일하고 각 줄 안의 공백을 정리합니다. 빈 줄은 문단 경계로 쓰입니다.
def normalize_prose(text:str)->str:
    # BEGIN D24-03-I02
    # TODO D24-03-I02 : CRLF·CR·가로 공백만 정리하며 문단 경계를 보존합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 줄바꿈을 모두 \n으로 통일합니다.
    # CRLF는 두 문자가 묶인 줄바꿈입니다. CR 한 글자부터 바꾸면 CRLF가 줄바꿈 두 개로 늘어납니다.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 같은 글자의 유니코드 표현을 통일합니다.
    text = apply_nfc(text)

    # 각 줄을 따로 정리합니다.
    cleaned_lines = []

    # split('\n')은 빈 줄도 빈 문자열로 남깁니다. append로 빈 줄까지 담아야 문단 경계가 유지됩니다.
    for line in text.split("\n"):
        # 연속된 공백이나 탭을 공백 한 칸으로 줄입니다.
        # [ \t]는 공백 또는 탭 한 글자이고, +는 그런 문자가 한 번 이상 이어지는 경우를 뜻합니다.
        line = re.sub(r"[ \t]+", " ", line)
        # 현재 줄의 앞뒤 공백을 없앱니다.
        cleaned_lines.append(line.strip())

    # 줄바꿈으로 다시 이어 붙이고, 전체 문자열의 양끝을 정리합니다.
    return "\n".join(cleaned_lines).strip()

    # END D24-03-I02
# 구절마다 정제 결과에 들어 있는지 살피면 정제 중에 빠진 숫자나 예외 조건을 찾을 수 있습니다.
def check_phrases(text:str,phrases:list[str])->dict:
    # BEGIN D24-03-I03
    # TODO D24-03-I03 : 핵심 구절 목록을 받아 부정어·숫자·예외의 존재 여부를 검사합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # phrases의 각 구절 p를 하나씩 검사합니다. 정제 본문에 그대로 들어 있으면 present, 빠졌으면 missing에 담습니다.
    return {'present':[p for p in phrases if p in text],
            'missing':[p for p in phrases if p not in text]}
    # END D24-03-I03
# 전후 길이와 규칙 버전은 변경을 설명하고, clean_hash는 정제 결과가 바뀌었는지 비교하는 데 쓰입니다.
def make_audit(original:str,cleaned:str,phrases:list[str])->dict:
    # BEGIN D24-03-I04
    # TODO D24-03-I04 : 정제 규칙 버전·전후 길이·해시·검사 결과를 출력합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # len()으로 전후 문자 수를 셉니다. clean_hash는 정제 문자열을 UTF-8 바이트로 바꾼 뒤 구한 SHA-256입니다.
    # 정제한 문자열에 같은 규칙을 다시 적용했을 때 결과가 같으면 idempotent가 True입니다.
    return {'cleaner_version':'nfc-prose-v1','chars_before':len(original),
            'chars_after':len(cleaned),'clean_hash':hashlib.sha256(cleaned.encode('utf-8')).hexdigest(),
            'phrase_check':check_phrases(cleaned,phrases),
            'idempotent':normalize_prose(cleaned)==cleaned}
    # END D24-03-I04
def main():
    # manifest에서 출석 안내 문서를 찾고, Markdown 파싱 결과 중 첫 번째 절을 골라 사용합니다.
    doc=by_doc('DOC-ATTEND');unit=parse_one(doc)['units'][0]
    original=unit['raw_text'];cleaned=normalize_prose(original)
    # 샘플에서 의미를 좌우하는 시간, 처리 결과, 예외 조건을 검사 대상으로 넘깁니다.
    audit=make_audit(original,cleaned,['10분','결석','사전 승인'])
    # 문서와 절의 ID를 정제 본문 및 검사 결과와 함께 outputs/clean_report.json에 저장합니다.
    out=save_json(HERE,'clean_report.json',{'doc_id':doc['doc_id'],
        'unit_id':unit['unit_id'],'clean_text':cleaned,'audit':audit})
    show_success('03 Intermediate',out)
