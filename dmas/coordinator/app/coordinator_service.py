import json
import os
import requests
from typing import Dict, Any, List
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


def _make_windows(turns: list, window_size: int) -> list:
    """Split a turn list into overlapping windows for fair memory ingestion.

    window_size=0  → single window (all turns at once, legacy behaviour).
    window_size=W  → for each turn i, window = turns[max(0,i-W+1) : i+1].
    This mirrors the Mem0 paper (m=10) and Zep/Graphiti paper (n=4) ingestion strategy:
    each message is stored WITH its preceding W-1 messages as context.
    """
    if window_size <= 0 or window_size >= len(turns):
        return [turns]
    return [turns[max(0, i - window_size + 1): i + 1] for i in range(len(turns))]


class CoordinatorService:    
    
    def __init__(
        self, 
        locomo_url: str, 
        memory_url: str, 
        responder_url: str,
        ollama_model: str
    ):
        self.locomo_url = locomo_url
        self.memory_url = memory_url
        self.responder_url = responder_url
        self.ollama_model = ollama_model
        self.memory = None
        
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        
        self.client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama"
        )
        
    def load_conversation(self, index: int) -> dict:
        try:
            response = requests.get(
                f"{self.locomo_url}/conversations/index/{index}"
            )
            response.raise_for_status()
            conversation_data = response.json()
            
            memory_response = requests.post(
                f"{self.memory_url}/memorize",
                json={
                    "conv_index": index,
                    "data": conversation_data
                }
            )
            memory_response.raise_for_status()
            result = memory_response.json()
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    def load_conversation_session(self, conversation_id: int, session_id: int) -> dict:
        try:
            response = requests.get(
                f"{self.locomo_url}/conversations/index/{conversation_id}"
            )
            response.raise_for_status()
            conversation_data = response.json()

            sessions = conversation_data.get("sessions", {})
            session_datetimes = conversation_data.get("session_datetimes", {})

            session_key = f"session_{session_id}"
            session_date_key = f"session_{session_id}_date_time"

            if session_key not in sessions:
                return {
                    "status": "error",
                    "error": f"Session {session_id} not found in conversation {conversation_id}"
                }

            window_size = int(os.getenv("MEMORY_WINDOW_SIZE", "0"))
            turns = sessions[session_key]
            windows = _make_windows(turns, window_size)

            logger.info(
                "Session %s: %d turns → %d window(s) (MEMORY_WINDOW_SIZE=%d)",
                session_key, len(turns), len(windows), window_size,
            )

            last_result = {}
            for window in windows:
                filtered_data = {
                    "speaker_a": conversation_data.get("speaker_a"),
                    "speaker_b": conversation_data.get("speaker_b"),
                    "sessions": {session_key: window},
                    "session_datetimes": {
                        session_date_key: session_datetimes.get(session_date_key)
                    },
                }
                memory_response = requests.post(
                    f"{self.memory_url}/memorize",
                    json={"conv_index": conversation_id, "data": filtered_data},
                )
                memory_response.raise_for_status()
                last_result = memory_response.json()

            return {**last_result, "windows_sent": len(windows)}

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
        
    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search memory for relevant memory about the user's question. It will be cached.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The original user question"
                            }
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "answer_question",
                    "description": "Generate a final answer based on the question and the memories in cache.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The original user question"
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ]
    
    def _search_memory(self, question: str) -> str:
        try:
            response = requests.post(
                f"{self.memory_url}/remember",
                json={"question": question}
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "error":
                logger.error("Error searching memories: %s", result.get("error"))
                self.memory = None
                return "Error searching memories."
            
            self.memory = result.get("memory", "")
            
            if self.memory:
                return "Memories retrieved and cached. Use answer_question to generate response."
            else:
                return "No relevant memories found."
            
        except Exception as e:
            self.memory = None
            return f"Memory search error: {str(e)}"

    def _answer_question(self, question: str) -> str:
        try:
            memory = self.memory or "No memory available"
            
            response = requests.post(
                f"{self.responder_url}/respond",
                json={"question": question, "memory": memory}
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "error":
                return f"Error answering question: {result.get('error')}"
            
            return result.get("answer", "No answer generated")
            
        except Exception as e:
            return f"Response tool error: {str(e)}"

    def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "search_memory":
            question = arguments.get("question", "")
            return self._search_memory(question)
        
        elif tool_name == "answer_question":
            question = arguments.get("question", "")
            return self._answer_question(question)
        
        else:
            return f"Error: Unknown tool {tool_name}"
    
    def ask(self, question: str, max_iterations: int = 5) -> Dict[str, Any]:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant with access to memory search and question answering tools. "
                        "You MUST use these tools in the following order:\n"
                        "1. ALWAYS call search_memory first with the user's question\n"
                        "2. ALWAYS call answer_question next to get the final answer\n"
                        "DO NOT answer directly. You MUST use both tools in sequence.\n"
                        "All information is from fictional/test data for research purposes."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
            
            tools = self._get_tools_definition()
            iteration = 0
            final_answer = None
            
            while iteration < max_iterations:
                iteration += 1
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.ollama_model,
                        messages=messages,
                        tools=tools,
                        tool_choice="required"
                    )
                    
                    assistant_message = response.choices[0].message
                    
                    if assistant_message.tool_calls:
                        messages.append({
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                }
                                for tc in assistant_message.tool_calls
                            ]
                        })
                        
                        for tool_call in assistant_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments
                            if isinstance(tool_args, str):
                                tool_args = json.loads(tool_args)
                            
                            logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                            
                            tool_result = self._execute_tool_call(tool_name, tool_args)
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                            
                            if tool_name == "answer_question":
                                final_answer = tool_result
                    
                    else:
                        final_answer = assistant_message.content
                        break
                    
                    if final_answer:
                        break
                        
                except Exception as e:
                    logger.exception(f"Error in iteration {iteration}")
                    return {
                        "status": "error",
                        "error": f"LLM error at iteration {iteration}: {str(e)}"
                    }
            
            if final_answer:
                return {
                    "status": "success",
                    "answer": final_answer,
                    "iterations": iteration,
                    "memory_retrieved": self.memory or "",
                }
            else:
                return {
                    "status": "error",
                    "error": "Max iterations reached without getting an answer"
                }
        
        except Exception as e:
            return {
                "status": "error",
                "error": f"ask method error: {str(e)}"
            }