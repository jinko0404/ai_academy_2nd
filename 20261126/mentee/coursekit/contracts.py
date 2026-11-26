"""Python 표준 라이브러리만으로 검사 가능한 계약. 값은 실측이 아니라 함수 검증에 사용."""
import re
from typing import Any

ALLOWED_TOOL_NAMES = frozenset({'rag_search','lookup_status'})
SOURCE_ID_PATTERN = re.compile(r'RPT-\d+')


def report_id_from(value: Any) -> int:
    """bool, 소수, 음수, 비정수 문자열을 거절합니다. 추측으로 반올림하지 않습니다."""
    if isinstance(value,bool) or not isinstance(value,(int,str)):
        raise ValueError('report_id는 양의 정수여야 합니다.')
    if isinstance(value,str) and (not value.isdecimal() or value.startswith('0')):
        raise ValueError('report_id 문자열은 선행 0 없는 양의 정수여야 합니다.')
    value = int(value)
    if value < 1:
        raise ValueError('report_id는 1 이상이어야 합니다.')
    return value


def search_request(query: str, category: str = 'ALL') -> tuple[str,str]:
    if not isinstance(query,str) or not query.strip() or len(query.strip())>500:
        raise ValueError('검색어는 1~500자여야 합니다.')
    from coursekit.settings import CATEGORIES
    category = category.upper().strip()
    if category not in {'ALL',*CATEGORIES}:
        raise ValueError('허용되지 않는 category입니다.')
    return query.strip(), category


def tool_call_request(name: str, args: dict, call_id: str) -> dict:
    if name not in ALLOWED_TOOL_NAMES:
        raise ValueError('미등록 Tool 이름입니다.')
    if not isinstance(args,dict) or not isinstance(call_id,str) or not call_id.strip():
        raise ValueError('인수는 dict이고 호출 ID는 빈 문자열이 아니어야 합니다.')
    return {'name':name,'args':args,'id':call_id}


def tool_result_message(call: dict, result: dict) -> dict:
    """ToolMessage 역할의 교육용 dict. 실제 LangChain ToolMessage 자체가 아닙니다."""
    if not isinstance(result,dict):
        raise ValueError('Tool 결과는 dict이어야 합니다.')
    if not call.get('id'):
        raise ValueError('Tool Call ID가 누락됐습니다.')
    return {'type':'ToolMessage','tool_call_id':call['id'],'name':call['name'],'content':result}


def validate_source_contract(result: dict, verified_source_ids: set[str]) -> bool:
    """출처 목록은 증거로 제공된 실제 DB Source ID에서만 선택해야 합니다."""
    status=result.get('status')
    sources=result.get('sources',[])
    if not isinstance(sources,list): raise ValueError('sources는 list여야 합니다.')
    ids=[x.get('source_id') for x in sources if isinstance(x,dict)]
    if len(ids)!=len(sources) or len(ids)!=len(set(ids)):
        raise ValueError('source_id 누락·중복')
    if status in {'insufficient_context','not_found'} and ids:
        raise ValueError('정보 부족에 근거 출처를 첨부할 수 없습니다.')
    if any(i not in verified_source_ids for i in ids):
        raise ValueError('검증된 Context 외부의 출처를 인용했습니다.')
    return True


def count_tool_calls(messages: list) -> dict:
    """Tool 선택 요청과 실제 ToolMessage를 별개로 집계합니다."""
    calls=[]; results=[]
    for m in messages:
        if isinstance(m,dict):
            calls += [c.get('id') for c in m.get('tool_calls',[]) if c.get('id')]
            if m.get('type')=='ToolMessage':results.append(m.get('tool_call_id'))
        else:
            calls += [c.get('id') for c in getattr(m,'tool_calls',[]) if c.get('id')]
            if m.__class__.__name__=='ToolMessage':results.append(getattr(m,'tool_call_id',None))
    return {'requested':len(calls),'executed_messages':len(results),'unmatched':sorted(set(calls)-set(results))}


def sanitize_event(event:dict)->dict:
    """질문 원문·API 키를 로그에서 제거하고 실행 메타데이터만 저장합니다."""
    permitted={'event','stage','status','tool_name','report_id','elapsed_ms','error_code','tool_call_id'}
    return {k:v for k,v in event.items() if k in permitted}
