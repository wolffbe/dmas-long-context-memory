import os
from fastapi import FastAPI, HTTPException
import uvicorn
from mem0 import Memory
import requests
import json

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": os.getenv("QDRANT_COLLECTION", "conversations"),
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
    },
}

m = Memory.from_config(config)

app = FastAPI(title="memory", version="1.0")


def extract_first_name(question: str) -> str:
    """
    Use Ollama to extract the first name from a question.
    Returns the extracted first name or empty string if none found.
    """
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        
        prompt = f"""Extract only the first name from the following question. 
If there is no first name mentioned, respond with "none".
Only respond with the first name or "none", nothing else.

Question: {question}

First name:"""
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 10
                }
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        first_name = result.get("response", "").strip().lower()
        
        if first_name == "none" or not first_name:
            return ""
        
        return first_name
        
    except Exception as e:
        print(f"Error extracting first name: {e}")
        return ""


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/memorize")
async def memorize(request: dict):
    try:
        conv_index = request.get("conv_index")
        data = request.get("data")
        
        if conv_index is None or not data:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: conv_index, data"
            )
                
        memories_added = 0
        for key, value in data.items():
            if key.startswith("session_") and not key.endswith("_date_time"):
                session_key = key
                turns = value
                
                if not isinstance(turns, list):
                    continue
                
                timestamp_key = f"{session_key}_date_time"
                timestamp = data.get(timestamp_key, "")
                
                for turn in turns:
                    if isinstance(turn, dict):
                        text = turn.get("text", "")
                        speaker = turn.get("speaker", "")
                        dia_id = turn.get("dia_id", "")
                        blip_caption = turn.get("blip_caption", "")
                        
                        if text:
                            memory_text = f"[{timestamp}] {speaker}: {text}"
                            
                            if blip_caption:
                                memory_text += f" [{blip_caption}]"
                            
                            m.add(memory_text, user_id=speaker.lower())
                            memories_added += 1
        
        return {
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: dict):
    try:
        prompt = request.get("prompt")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt field")
        
        first_name = extract_first_name(prompt)
        
        user_id = first_name if first_name else "default"
        
        memories = m.search(query=prompt, user_id=user_id)
        
        if memories:
            context_parts = [mem.get("memory", "") for mem in memories]
            context = " ".join(context_parts)
        else:
            context = ""
        
        return {
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)