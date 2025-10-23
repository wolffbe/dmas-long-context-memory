import os
from fastapi import FastAPI, HTTPException
import uvicorn
from mem0 import Memory
import requests
import json
from datetime import datetime, timezone
from typing import Optional
import asyncio

MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "mem0").lower()

if MEMORY_BACKEND == "graphiti":
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    
    llm_config = LLMConfig(
        api_key="abc",
        model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        small_model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
    )
    llm_client = OpenAIClient(config=llm_config)
    
    graphiti = Graphiti(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "password"),
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="abc",
                embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                embedding_dim=int(os.getenv("EMBEDDING_DIMS", "768")),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
            )
        ),
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )
    print(f"✓ Initialized Graphiti memory backend")
else:
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": os.getenv("QDRANT_COLLECTION", "conversations"),
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "embedding_model_dims": int(os.getenv("EMBEDDING_DIMS", "768")),
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
    print(f"✓ Initialized Mem0 memory backend")

app = FastAPI(title="memory", version="1.0")


@app.on_event("startup")
async def startup_event():
    if MEMORY_BACKEND == "graphiti":
        print("Initializing Graphiti indices and constraints...")
        try:
            await graphiti.build_indices_and_constraints()
            print("✓ Graphiti indices and constraints initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Graphiti indices: {e}")
            print("This is normal if indices already exist")


@app.on_event("shutdown")
async def shutdown_event():
    if MEMORY_BACKEND == "graphiti":
        print("Closing Graphiti connection...")
        await graphiti.close()
        print("✓ Graphiti connection closed")


def extract_first_name(question: str) -> str:
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
    return {
        "status": "healthy",
        "backend": MEMORY_BACKEND
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
        
        if MEMORY_BACKEND == "graphiti":
            return await memorize_graphiti(conv_index, data)
        else:
            return await memorize_mem0(conv_index, data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def memorize_mem0(conv_index: int, data: dict):
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
    
    return {"status": "success"}


async def memorize_graphiti(conv_index: int, data: dict):
    episodes_added = 0
    
    for key, value in data.items():
        if key.startswith("session_") and not key.endswith("_date_time"):
            session_key = key
            turns = value
            
            if not isinstance(turns, list):
                continue
            
            timestamp_key = f"{session_key}_date_time"
            timestamp = data.get(timestamp_key, "")
            
            if not timestamp:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing timestamp for {session_key}"
                )
            
            reference_time = None
            try:
                reference_time = datetime.strptime(timestamp, "%I:%M %p on %d %B, %Y")
                reference_time = reference_time.replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    reference_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    reference_time = reference_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid timestamp format for {session_key}: '{timestamp}'. Expected formats: '7:55 pm on 9 June, 2023' or 'YYYY-MM-DD HH:MM:SS'"
                    )
            
            episode_parts = []
            speakers = set()
            
            for turn in turns:
                if isinstance(turn, dict):
                    text = turn.get("text", "")
                    speaker = turn.get("speaker", "")
                    blip_caption = turn.get("blip_caption", "")
                    
                    if text:
                        speakers.add(speaker)
                        turn_text = f"[{timestamp}] {speaker}: {text}"
                        if blip_caption:
                            turn_text += f" [Image: {blip_caption}]"
                        episode_parts.append(turn_text)
            
            if episode_parts:
                episode_body = "\n".join(episode_parts)
                
                speaker_list = ", ".join(sorted(speakers))
                description = f"Conversation session with participants: {speaker_list}"
                if timestamp:
                    description += f" at {timestamp}"
                
                await graphiti.add_episode(
                    name=f'Conversation {conv_index} - {session_key}',
                    episode_body=episode_body,
                    source="conversation",
                    source_description=description,
                    reference_time=reference_time,
                )
                
                episodes_added += 1
    
    return {"status": "success"}


@app.post("/query")
async def query(request: dict):
    try:
        prompt = request.get("prompt")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt field")
        
        if MEMORY_BACKEND == "graphiti":
            return await query_graphiti(prompt)
        else:
            return await query_mem0(prompt)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def query_mem0(prompt: str):
    first_name = extract_first_name(prompt)
    user_id = first_name if first_name else "default"
    
    memories = m.search(query=prompt, user_id=user_id)
    
    result = {"status": "success"}
    
    if memories:
        context_parts = [mem.get("memory", "") for mem in memories]
        context = " ".join(context_parts)
        if context:
            result["context"] = context
    
    return result


async def query_graphiti(prompt: str):
    results = await graphiti.search(prompt)
    
    if results and len(results) > 0:
        center_node_uuid = results[0].source_node_uuid
        results = await graphiti.search(prompt, center_node_uuid=center_node_uuid)
    
    result = {"status": "success"}
    
    if results:
        facts = []
        for r in results:
            facts.append(r.fact)
        
        context = " ".join(facts)
        if context:
            result["context"] = context
    
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)