from pydantic import BaseModel

class ResponseRequest(BaseModel):
    question: str
    backend: str  # "mem0" | "graphiti" | "rag" | "cognee" | "full_context"