import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import ollama


app = FastAPI(title="responder", version="1.0")


class RespondRequest(BaseModel):
    question: str
    context: str


class Responder:
    def __init__(self, model: str):
        self.model = model
    
    def respond(self, question: str, context: str) -> dict:
        try:
            if not context:
                context = "No relevant information available."
            
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_ctx": int(os.getenv("NUM_CTX", "4096"))}
            )
            
            answer = response["message"]["content"]
            
            return {
                "status": "success",
                "answer": answer
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


responder = Responder(
    model=os.getenv("MODEL", "llama3.2:8b")
)


@app.get("/health")
async def health():
    try:
        ollama.list()
        return {"status": "healthy", "model": responder.model}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/respond")
async def respond(request: RespondRequest):
    try:
        result = responder.respond(request.question, request.context)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)