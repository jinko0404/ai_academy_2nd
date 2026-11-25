"""Day28: 질문 입력과 Gold 레이블을 물리적으로 분리하는 평가 데이터 계약.

Target에는 question_id, question, category, history만 전달합니다.
Gold를 검색 입력·Prompt에 넣는 것은 시험 정답 유출이므로 금지합니다.
"""
from pathlib import Path
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TARGET_KEYS = frozenset({'gold_chunk_ids','graded_relevance','reference_answer','answerable','gold','expected_answer','label'})


def read_json(filename: str):
    path = ROOT/'data'/filename
    return json.loads(path.read_text(encoding='utf-8'))


def read_questions():
    questions=read_json('questions.json')
    if not isinstance(questions,list) or not questions:
        raise ValueError('질문 목록이 비었거나 배열이 아닙니다.')
    keys=[q['question_id'] for q in questions]
    if len(set(keys))!=len(keys):
        raise ValueError('question_id가 중복됩니다.')
    return questions


def read_gold():
    data=read_json('gold_labels.json')
    return data['gold'], data['graded_relevance'], data['dataset_version']


def target_inputs(example):
    """정답을 모른 채 현재 서비스를 실행하는 입력만 새 dict로 구성."""
    safe_keys=('question_id','question','category','history')
    if not all(k in example for k in safe_keys):
        raise ValueError('Target 입력 필수 필드 누락')
    return {key:example[key] for key in safe_keys}


def assert_no_gold_leakage(request):
    found=FORBIDDEN_TARGET_KEYS.intersection(request)
    if found:raise ValueError('Gold Label/Reference Answer가 Target 입력에 포함됨: '+','.join(sorted(found)))


def validate_dataset(questions, gold, grades, corpus):
    if set(gold)!=set(q['question_id'] for q in questions):
        raise ValueError('평가 질문과 Gold 질문 ID 불일치')
    if set(grades)!=set(gold):raise ValueError('관련성 등급 질문 ID 불일치')
    docs={d['chunk_id']:d for d in corpus}
    for q in questions:
        qid=q['question_id']; ids=gold[qid]
        if len(set(ids))!=len(ids):raise ValueError(qid+': Gold 중복')
        if q['answerable'] != bool(ids):raise ValueError(qid+': answerable 불일치')
        if {cid for cid,grade in grades[qid].items() if grade>0}!=set(ids):
            raise ValueError(qid+': graded positive IDs / Gold 불일치')
        for graded_id,graded_score in grades[qid].items():
            document=docs.get(graded_id)
            if document is None or document['category']!=q['category'] or not document['is_current'] or not document['ready'] or document['access_level']!='public':
                raise ValueError(qid+': 판정 문서가 허용 범위 밖에 있음')
            if type(graded_score) is not int or not 0<=graded_score<=3:raise ValueError(qid+': 등급 범위 0~3')
        for cid in ids:
            d=docs.get(cid)
            if d is None:raise ValueError(qid+': 없는 Gold ID '+cid)
            if d['category']!=q['category'] or not d['is_current'] or not d['ready'] or d['access_level']!='public':
                raise ValueError(qid+': Gold가 접근 가능한 현재 문서가 아님: '+cid)
            if type(grades[qid][cid]) is not int or not 1<=grades[qid][cid]<=3:
                raise ValueError(qid+': 잘못된 관련성 등급')
    return {'questions':len(questions),'answerable':sum(bool(gold[q['question_id']]) for q in questions),
            'no_answer':sum(not gold[q['question_id']] for q in questions),'gold_ids':sum(map(len,gold.values()))}
