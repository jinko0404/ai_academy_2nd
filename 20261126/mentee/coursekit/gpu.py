"""GPU-only E5 + CrossEncoder. CUDA가 없으면 실패하고 CPU로 폴백하지 않습니다."""
from functools import lru_cache
import math
from coursekit.settings import settings


def require_cuda()->dict:
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Day29 실습은 CUDA GPU가 필수입니다. torch.cuda.is_available()=False; CPU 폴백 금지.')
    device=settings().cuda_device
    i=int(device.split(':')[1])
    if i>=torch.cuda.device_count():
        raise RuntimeError(f'CUDA_DEVICE={device}가 존재하지 않습니다.')
    return {'device':device,'name':torch.cuda.get_device_name(i),'count':torch.cuda.device_count()}


@lru_cache(maxsize=1)
def e5_model():
    require_cuda()
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(settings().embedding_model,device=settings().cuda_device)
    if not str(model.device).startswith('cuda'):
        raise RuntimeError('E5가 CUDA 장치에 올라가지 않았습니다.')
    return model


def encode_query(query:str)->list[float]:
    """11/20의 passage: overview Embedding과 같은 모델, query: prefix 사용."""
    model=e5_model()
    text='query: '+query
    tokens=model.tokenizer(text,add_special_tokens=True,truncation=False)['input_ids']
    if len(tokens)>int(model.max_seq_length):
        raise ValueError('E5 토큰 한도 초과. 질문을 줄이세요; 조용한 truncation 금지.')
    vec=model.encode([text],normalize_embeddings=True,show_progress_bar=False,batch_size=1)[0]
    arr=[float(x) for x in vec]
    if len(arr)!=384 or any(not math.isfinite(x) for x in arr):
        raise RuntimeError('E5 벡터 차원/NaN/Inf 검사 실패')
    return arr


@lru_cache(maxsize=1)
def cross_encoder():
    require_cuda()
    from sentence_transformers import CrossEncoder
    model=CrossEncoder(settings().reranker_model,device=settings().cuda_device)
    if not str(model.device).startswith('cuda'):
        raise RuntimeError('CrossEncoder가 CUDA 장치에 올라가지 않았습니다.')
    return model


def rerank(query:str,rows:list[dict])->list[dict]:
    if not rows:return []
    model=cross_encoder()
    pairs=[(query, (row['title']+'\n'+row['overview']+'\n'+row['content'][:900])) for row in rows]
    scores=model.predict(pairs,batch_size=4,show_progress_bar=False)
    output=[{**row,'rerank_score':float(score)} for row,score in zip(rows,scores)]
    return sorted(output,key=lambda x:(-x['rerank_score'],x['report_id']))
