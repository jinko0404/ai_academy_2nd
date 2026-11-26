"""실습에서 제공하는 로컬 FastAPI 시작 코드. 구현할 TODO는 tasks.py에 있습니다.

POST 요청에서 질문을 읽고 Agent를 실행한 뒤, 전달받은 함수로 응답을 검증하고 정리합니다.
"""
from coursekit.agent import run_agent

def make_api(api_response):
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel,Field
    # FastAPI는 Ask 모델에 따라 JSON 본문의 question이 1~1000자인지 확인합니다.
    class Ask(BaseModel):
        question:str=Field(min_length=1,max_length=1000)
    app=FastAPI(title='Day29 Report Agent (Local Only)')
    # 이 데코레이터가 POST /api/agent/ask 요청을 아래 ask() 함수에 연결합니다.
    @app.post('/api/agent/ask')
    def ask(body:Ask):
        # 요청마다 run_agent()가 실행되고, 결과는 main.py가 넘긴 api_response()를 거칩니다.
        try:return api_response(run_agent(body.question))
        # 예외 종류를 HTTP 상태 코드로 바꾸고, 사용자에게 고정된 오류 메시지를 반환합니다.
        except ValueError:raise HTTPException(422,'잘못된 입력')
        except PermissionError:raise HTTPException(403,'접근 정책 위반')
        except RuntimeError as exc:raise HTTPException(502,'Agent 결과/출처 검증 실패') from exc
        except Exception as exc:raise HTTPException(503,'Agent Tool 실행 실패') from exc
    return app
