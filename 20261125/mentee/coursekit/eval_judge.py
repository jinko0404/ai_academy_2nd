"""선택적 Live LLM-as-a-Judge. 기본 실습에서 자동 호출되지 않습니다.

주의: 이 Judge는 보조 판정이며 사람 검수를 대체하지 않습니다. 질문/Context/Claim에
API Key 또는 개인정보를 포함하지 않는다는 조건 아래 명시적으로 호출합니다.
"""
from typing import Literal
from pydantic import BaseModel,Field,ConfigDict

class ClaimAssessment(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str=Field(min_length=1,max_length=500)
    supported:bool
    evidence:str=Field(max_length=700,description='주어진 Context 중 판단 근거. 없는 정보는 만들지 않음')

class JudgeReview(BaseModel):
    model_config=ConfigDict(extra='forbid')
    claims:list[ClaimAssessment]
    relevance:Literal['높음','보통','낮음']
    correctness:Literal['정답','부분정답','오답','판정 불가']
    needs_human_review:bool


def create_judge_chain():
    """import/객체 생성과 API 호출 시점을 구분. .invoke() 때 실제 네트워크 호출."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from coursekit.config import require_setting
    model_id=require_setting('LLM_MODEL')
    require_setting('OPENAI_API_KEY')
    prompt=ChatPromptTemplate.from_messages([
        ('system','당신은 RAG 답변 품질 보조 평가자입니다. 현재 Context의 사실만 기준으로 각 Claim 지원 여부를 평가하세요. '
                  '확인이 어려우면 사람이 추가 검토하도록 표시하세요. JSON Schema를 준수하고 근거를 인용하세요.'),
        ('human','질문: {question}\n\nContext: {context}\n\n평가할 Claim 목록: {claims}\n\n참고 답안(정답성 비교용): {reference}')
    ])
    return prompt | ChatOpenAI(model=model_id,temperature=0).with_structured_output(JudgeReview)


def judge_case_live(case,reference=''):
    chain=create_judge_chain()
    claims=[x['text'] for x in case['claims']]  # 사람의 supported 정답은 Judge에 전달하지 않음
    if not claims:raise ValueError('Claim이 없는 사례는 Claim Judge로 강제 채점하지 않습니다.')
    result=chain.invoke({'question':case['question'],'context':case['context'],
                        'claims':str(claims),'reference':reference})
    if len(result.claims)!=len(claims):raise ValueError('Judge Claim 개수 불일치. 사람이 검토하세요.')
    return result
