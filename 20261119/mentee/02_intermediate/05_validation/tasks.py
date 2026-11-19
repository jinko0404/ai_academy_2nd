# 실습 목표 — 정제 본문의 문자 위치를 기준으로 커버리지를 구하고, 누락 구간과 중복 청크 ID를 찾습니다.
# 전체 흐름 — 20자 예제 준비 → 각 범위 확인 → 덮은 위치와 오류 계산 → 세 예제의 예상 비율 대조 → outputs/coverage_report.json 저장.
"""실습 05 Intermediate · 설명용 20자 경계 세트."""
from pathlib import Path
import sys
# 현재 실습 폴더를 기준으로 공통 coursekit을 불러오고, 이 폴더 아래에 결과를 저장합니다.
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import save_json,show_success
from coursekit.processing import covered_positions


# Span은 (시작, 끝)으로 나타낸 문자 범위입니다. 시작은 포함하고 끝은 제외합니다.
def validate_ranges(length:int,spans:list[tuple[int,int]])->None:
    # BEGIN D24-05-I01
    # TODO D24-05-I01 : 청크별 Span이 기준 본문의 범위 안인지 검증합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 각 범위 (a, b)에서 시작은 포함하고 끝은 제외합니다. (0, 8)은 인덱스 0부터 7까지입니다.
    for a,b in spans:
        # 음수 시작점, 길이가 0인 범위, 본문 끝을 넘는 범위는 ValueError로 처리합니다.
        if not 0<=a<b<=length:raise ValueError('invalid range')
    # END D24-05-I01
# 예를 들어 (0, 8)과 (6, 14)는 함께 14개 위치를 덮습니다. 겹친 두 문자를 한 번씩만 셉니다.
def coverage(length:int,spans:list[tuple[int,int]])->float|None:
    # BEGIN D24-05-I02
    # TODO D24-05-I02 : 합집합으로 덮은 문자 수와 Coverage를 계산합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 본문이 비어 있으면 기준 문자 수가 0이므로 비율을 None으로 표시합니다.
    if length==0:return None
    validate_ranges(length,spans)
    # covered_positions는 덮은 고유 문자 수와 누락 위치 목록을 함께 돌려줍니다.
    # 여기서는 비율에 필요한 문자 수만 n에 담고, 누락 위치 목록은 _에 받습니다.
    n,_=covered_positions(spans,length)
    # 예를 들어 20개 위치 중 18개를 덮으면 18 / 20 = 0.9, 즉 90%입니다.
    return n/length
    # END D24-05-I02
# Gap은 아무 청크도 덮지 않은 연속 문자 위치입니다. 같은 ID의 반복도 별도로 확인합니다.
def detect_gaps_and_duplicates(length:int,spans:list[tuple[int,int]],ids:list[str])->dict:
    # BEGIN D24-05-I03
    # TODO D24-05-I03 : 빠진 위치의 연속 구간 및 중복 Chunk ID를 구합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 여기서는 누락된 인덱스 목록만 사용합니다. 인덱스 8과 9가 연달아 빠지면 (8, 10)으로 묶습니다.
    _,missing=covered_positions(spans,length)
    gaps=[]
    for pos in missing:
        # gaps[-1]은 마지막 누락 구간입니다. 현재 위치가 직전 구간의 끝에 이어지면 끝을 한 칸 늘립니다.
        if gaps and gaps[-1][1]==pos:gaps[-1]=(gaps[-1][0],pos+1)
        else:gaps.append((pos,pos+1))
    # seen에는 이미 나온 ID를 담습니다. 같은 ID를 다시 만나면 dups에 추가합니다.
    seen=set();dups=[]
    for cid in ids:
        if cid in seen:dups.append(cid)
        seen.add(cid)
    # set(dups)로 중복된 ID 이름을 한 번씩만 남기고, sorted로 순서를 일정하게 맞춥니다.
    return {'gaps':gaps,'duplicate_ids':sorted(set(dups))}
    # END D24-05-I03
# Coverage가 1.0이어도 같은 청크를 두 번 넣었을 수 있습니다. 비율과 검증 통과 여부를 함께 확인합니다.
def make_report(length:int,spans:list[tuple[int,int]],ids:list[str])->dict:
    # BEGIN D24-05-I04
    # TODO D24-05-I04 : 통과·오류·반복량·Coverage를 구분한 보고서를 작성합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    validate_ranges(length,spans)
    covered,_=covered_positions(spans,length)
    errors=detect_gaps_and_duplicates(length,spans,ids)
    # 범위 길이 합에서 고유 문자 수를 빼면 여러 번 덮인 횟수가 나옵니다. 정상적인 겹침도 이 값에 포함됩니다.
    # valid는 누락 구간과 중복 ID 목록이 둘 다 비어 있을 때 True입니다.
    return {'coverage':coverage(length,spans), 'covered_characters':covered,
            'repeat_extra_chars':sum(b-a for a,b in spans)-covered,
            'errors':errors,'valid':not errors['gaps'] and not errors['duplicate_ids']}
    # END D24-05-I04
def main():
    # 예제 한 건은 (범위 목록, ID 목록)입니다. 정상 겹침, 문자 두 개 누락, 같은 ID 반복을 차례로 비교합니다.
    samples={'overlap':([(0,8),(6,14),(12,20)],['A','B','C']),
             'missing':([(0,8),(10,20)],['A','B']),
             'duplicate':([(0,8),(0,8),(8,20)],['A','A','C'])}
    # 각 parts는 (범위 목록, ID 목록)입니다. *parts가 두 값을 펼쳐 make_report의 인자로 전달합니다.
    # 딕셔너리 표현식은 예제 이름을 키로 두고, make_report가 반환한 결과를 값으로 저장합니다.
    reports={name:make_report(20,*parts) for name,parts in samples.items()}
    # 세 예제의 커버리지가 100%, 90%, 100%인지 확인합니다. 통과 여부는 각 결과의 valid를 봅니다.
    assert [reports[n]['coverage'] for n in samples]==[1.0,.9,1.0]
    out=save_json(HERE,'coverage_report.json',reports)
    show_success('05 Intermediate',out)
