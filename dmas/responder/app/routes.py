import os
# Import BEFORE the service — patches openai chat/embeddings/responses
# `create` to inject `metadata.tags` so litellm tags the trace.
from app import langfuse_tags  # noqa: F401

from fastapi import FastAPI, HTTPException

from app.responder_service import ResponderService
from app.models import ResponseRequest

app = FastAPI(title="responder", version="1.0")

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
        result = responder.respond(request.question, request.backend)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))