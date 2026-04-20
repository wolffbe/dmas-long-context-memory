import importlib
import os
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="memory", version="1.0")

_BACKEND_REGISTRY = {
    "mem0":         ("app.services.mem0_service",         "Mem0Service"),
    "graphiti":     ("app.services.graphiti_service",     "GraphitiService"),
    "rag":          ("app.services.rag_service",          "RagService"),
    "full_context": ("app.services.full_context_service", "FullContextService"),
}
_DEFAULT_BACKEND = "mem0"

def _create_backend(name: str):
    module_path, class_name = _BACKEND_REGISTRY.get(name, _BACKEND_REGISTRY[_DEFAULT_BACKEND])
    module = importlib.import_module(module_path)
    logger.info("Loading memory backend: %s", class_name)
    return getattr(module, class_name)()

memory_backend = _create_backend(os.getenv("MEMORY_BACKEND", "").lower())

def get_mem_backend() -> Any:
    return memory_backend

class RememberRequest(BaseModel):
    question: str

@app.post("/memorize")
def memorize(
    request: Dict[str, Any],
    backend: Any = Depends(get_mem_backend)
):
    conv_index = request.get("conv_index")
    data = request.get("data")
    
    if conv_index is None or not data:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: conv_index, data"
        )
    
    memorize_func = getattr(backend, "memorize_conversation", None)
    if callable(memorize_func):
        return memorize_func(conv_index, data)
    
    raise HTTPException(status_code=500, detail="Backend does not support memorize_conversation")

@app.post("/remember")
def remember(
    request: RememberRequest,
    backend: Any = Depends(get_mem_backend)
):
    if not request.question:
        logger.error("Missing question!")
        raise HTTPException(status_code=400, detail="Missing question")
    
    remember_func = getattr(backend, "remember", None)
    
    if callable(remember_func):
        try:
            memories = remember_func(request.question)
        except Exception as e:
            logger.exception("Error calling remember_func")
            raise
        
        if not memories:
            logger.info("No memories found, returning empty")
            return {
                "status": "success",
                "memory": ""
            }
        
        memory_text = "\n\n".join(memories)
        logger.info("Joined %d memories into text (length: %d)", len(memories), len(memory_text))
        
        return {
            "status": "success",
            "memory": memory_text
        }
    
    raise HTTPException(status_code=500, detail="Backend does not support remember")

@app.get("/health")
def health():
    return {"status": "ok"}