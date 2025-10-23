import os
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import ollama
import json


app = FastAPI(title="coordinator", version="1.0")


class AskRequest(BaseModel):
    question: str


class Coordinator:    
    def __init__(self, locomo_url: str, memory_url: str, responder_url: str, model: str):
        self.locomo_url = locomo_url
        self.memory_url = memory_url
        self.responder_url = responder_url
        self.model = model
   
    def load_conversation(self, conv_index: int) -> dict:        
        try:
            response = requests.get(f"{self.locomo_url}/conversations/index/{conv_index}")
           
            if response.status_code == 404:
                return {
                    "status": "error",
                    "conversation_index": conv_index,
                    "error": "Conversation not found"
                }
           
            response.raise_for_status()
            conversation_data = response.json()
           
            mem_payload = {
                "conv_index": conv_index,
                "data": conversation_data
            }
           
            mem_response = requests.post(
                f"{self.memory_url}/memorize",
                json=mem_payload,
                timeout=30
            )
            mem_response.raise_for_status()
           
            return {
                "status": "success",
                "conversation_index": conv_index
            }
                   
        except Exception as e:
            return {
                "status": "error",
                "conversation_index": conv_index,
                "error": str(e)
            }
   
    def load_all_conversations(self) -> dict:
        results = []
        for conv_idx in range(10):
            result = self.load_conversation(conv_idx)
            results.append(result)
                       
        return {
            "status": "success",
            "total": 10,
            "results": results
        }
    
    def _define_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "Query memory for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "What to ask memory"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "respond",
                    "description": "Send question and context to responder for final answer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The user's question"
                            },
                            "context": {
                                "type": "string",
                                "description": "Context from memory"
                            }
                        },
                        "required": ["question", "context"]
                    }
                }
            }
        ]
    
    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        if tool_name == "remember":
            prompt = arguments["prompt"]
            try:
                response = requests.post(
                    f"{self.memory_url}/query",
                    json={"prompt": prompt},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "error":
                    return {"error": result.get("error")}
                
                context = result.get("context", "")
                if not context:
                    return {"context": "No relevant information found."}
                
                return {"context": context}
            except Exception as e:
                return {"error": str(e)}
        
        elif tool_name == "respond":
            question = arguments["question"]
            context = arguments["context"]
            try:
                response = requests.post(
                    f"{self.responder_url}/respond",
                    json={
                        "question": question,
                        "context": context
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "error":
                    return {"error": result.get("error")}
                
                return {"answer": result.get("answer")}
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "Unknown tool"}
    
    def ask_question(self, question: str) -> dict:
        try:
            prompt = f"""The user wants to know: {question}

You have the following tools:
- remember: Query memory for relevant information
- respond: Send question and context to responder for final answer

First use remember to get context, then use respond to get the final answer."""

            messages = [{"role": "user", "content": prompt}]
            tools = self._define_tools()
            final_answer = None
            
            for iteration in range(5):
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    options={"num_ctx": 512}
                )
                
                assistant_message = response["message"]
                messages.append(assistant_message)
                
                if not assistant_message.get("tool_calls"):
                    if final_answer:
                        return {
                            "status": "success",
                            "answer": final_answer
                        }
                    return {
                        "status": "error",
                        "error": "No answer generated"
                    }
                
                for tool_call in assistant_message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                    
                    tool_result = self._call_tool(tool_name, arguments)
                    
                    if tool_name == "respond" and "answer" in tool_result:
                        final_answer = tool_result.get("answer")
                    
                    messages.append({"role": "tool", "content": json.dumps(tool_result)})
                
                if final_answer:
                    return {
                        "status": "success",
                        "answer": final_answer
                    }
            
            return {
                "status": "error",
                "error": "Max iterations reached"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


coordinator = Coordinator(
    locomo_url=os.getenv("API_URL", "http://locomo:8000"),
    memory_url=os.getenv("MEMORY_URL", "http://memory:8005"),
    responder_url=os.getenv("RESPONDER_URL", "http://responder:8006"),
    model=os.getenv("MODEL", "llama3.2:3b")
)


@app.get("/health")
async def health():
    try:
        requests.get(f"{coordinator.locomo_url}/health", timeout=5).raise_for_status()
        requests.get(f"{coordinator.memory_url}/health", timeout=5).raise_for_status()
        requests.get(f"{coordinator.responder_url}/health", timeout=5).raise_for_status()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/load_conversations")
async def load_conversations(index: Optional[int] = None):
    try:
        if index is not None:
            return coordinator.load_conversation(index)
        else:
            return coordinator.load_all_conversations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load_conversation/index/{index}")
async def load_conversation_by_index(index: int):
    try:
        if index < 0 or index > 9:
            raise HTTPException(status_code=400, detail="Index must be between 0 and 9")
        return coordinator.load_conversation(index)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(request: AskRequest):
    try:
        result = coordinator.ask_question(request.question)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)