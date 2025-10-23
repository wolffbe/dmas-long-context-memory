import os
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
import uvicorn
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO


class MemoryAgent:
    
    def __init__(self):
        persist_directory = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.text_model = SentenceTransformer('all-mpnet-base-v2')
        self.image_model = SentenceTransformer('clip-ViT-B-32')
        
        self.collection_name = os.getenv("CHROMA_COLLECTION", "conversations")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Turn-level conversation embeddings"}
        )
        
        print(f"✓ Connected to ChromaDB: {self.collection_name}")
        print(f"✓ Persist directory: {persist_directory}")
        print(f"✓ Current vector count: {self.collection.count()}")
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"✗ Failed to download image {url}: {e}")
            return None
    
    def memorize(self, conv_index: int, data: Any) -> dict:
        try:
            sessions = data.get("sessions", {})
            sample_id = data.get("sample_id", f"conv_{conv_index}")
            speaker_a = data.get("speaker_a", "")
            speaker_b = data.get("speaker_b", "")
            session_datetimes = data.get("session_datetimes", {})
            
            embeddings_added = 0
            
            for session_key, turns in sessions.items():
                if not isinstance(turns, list):
                    continue
                
                session_datetime = session_datetimes.get(f"{session_key}_date_time", "")
                
                for turn_idx, turn in enumerate(turns):
                    if isinstance(turn, dict):
                        speaker = turn.get("speaker", "")
                        dia_id = turn.get("dia_id", f"{session_key}_{turn_idx}")
                        text = turn.get("text", "")
                        img_url = turn.get("img_url")
                        
                        if text:
                            text_embedding = self.text_model.encode(text).tolist()
                            
                            doc_id = f"{sample_id}_{session_key}_{dia_id}_text"
                            metadata = {
                                "type": "turn_text",
                                "sample_id": sample_id,
                                "session": session_key,
                                "session_datetime": session_datetime,
                                "dia_id": dia_id,
                                "speaker_a": speaker_a,
                                "speaker_b": speaker_b
                            }
                            
                            self.collection.upsert(
                                ids=[doc_id],
                                embeddings=[text_embedding],
                                metadatas=[metadata],
                                documents=[f"{speaker}: {text}"]
                            )
                            embeddings_added += 1
                        
                        if img_url:
                            image = self._download_image(img_url)
                            if image:
                                image_embedding = self.image_model.encode(image).tolist()
                                
                                doc_id = f"{sample_id}_{session_key}_{dia_id}_image"
                                metadata = {
                                    "type": "turn_image",
                                    "sample_id": sample_id,
                                    "session": session_key,
                                    "session_datetime": session_datetime,
                                    "dia_id": dia_id,
                                    "img_url": img_url
                                }
                                
                                caption = f"{speaker} shared an image"
                                if text:
                                    caption += f": {text}"
                                
                                self.collection.upsert(
                                    ids=[doc_id],
                                    embeddings=[image_embedding],
                                    metadatas=[metadata],
                                    documents=[caption]
                                )
                                embeddings_added += 1
            
            print(f"✓ Memorized {sample_id}: {embeddings_added} turns")
            
            return {
                "status": "success",
                "message": f"Memorized {embeddings_added} turns",
                "embeddings_added": embeddings_added,
                "vector_count": self.collection.count()
            }
        
        except Exception as e:
            print(f"✗ Error memorizing: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_stats(self) -> dict:
        try:
            return {
                "collection_name": self.collection_name,
                "total_vectors": self.collection.count(),
                "text_model": "all-mpnet-base-v2",
                "image_model": "clip-ViT-B-32",
                "embedding_dimension": 768
            }
        except Exception as e:
            return {"error": str(e)}


app = FastAPI(title="memory", version="1.0")
memory_agent = MemoryAgent()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vector_count": memory_agent.collection.count()
    }


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
        
        result = memory_agent.memorize(conv_index, data)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(request: dict):
    try:
        prompt = request.get("prompt")
        max_tokens = request.get("max_tokens", 3000)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt field")
        
        query_embedding = memory_agent.text_model.encode(prompt).tolist()
        
        results = memory_agent.collection.query(
            query_embeddings=[query_embedding],
            n_results=50
        )
        
        documents = results.get("documents", [[]])[0]
        
        if not documents:
            return {
                "status": "success",
                "context": ""
            }
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4
        
        for doc in documents:
            if total_chars + len(doc) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    context_parts.append(doc[:remaining])
                break
            context_parts.append(doc)
            total_chars += len(doc)
        
        context = " ".join(context_parts)
        
        return {
            "status": "success",
            "context": context
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    return memory_agent.get_stats()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)