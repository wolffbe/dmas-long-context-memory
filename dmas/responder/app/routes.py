import os
from shared import otel_init

otel_init.init("responder")
otel_init.instrument_httpx()
otel_init.instrument_requests()

from fastapi import FastAPI, HTTPException

from app.responder_service import ResponderService
from app.models import ResponseRequest

app = FastAPI(title="responder", version="1.0")
otel_init.instrument_fastapi(app)

responder = ResponderService(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    memory_url=os.getenv("MEMORY_URL", "http://toxiproxy:18005"),
)

@app.get("/health")
async def health():
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        return {"status": "healthy", "model": responder.model}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/respond")
async def respond(request: ResponseRequest):
    try:
        result = responder.respond(request.question, request.backend,
                                   request.session_date, request.trace_id,
                                   request.session_id,
                                   request.conv_index, request.mode)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))