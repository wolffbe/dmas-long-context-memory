from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
from pathlib import Path
import os
import urllib.request

app = FastAPI(title="locomo", version="1.0")

class DialogTurn(BaseModel):
    speaker: str
    dia_id: str
    text: str
    img_url: Optional[Any] = None
    blip_caption: Optional[Any] = None

class Question(BaseModel):
    question: str
    answer: Optional[Any] = None
    adversarial_answer: Optional[Any] = None
    category: Optional[Any] = None
    evidence: Optional[List[str]] = None
    
    def get_answer(self) -> str:
        if self.answer is not None:
            return str(self.answer)
        elif self.adversarial_answer is not None:
            return str(self.adversarial_answer)
        return ""

class EventSummary(BaseModel):
    pass

class Conversation(BaseModel):
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: Dict[str, List[DialogTurn]]
    session_datetimes: Dict[str, str]
    observations: Optional[Dict[str, Any]] = None
    session_summaries: Optional[Dict[str, Any]] = None
    event_summary: Optional[Dict[str, Any]] = None
    qa: Optional[List[Question]] = None

class ConversationStats(BaseModel):
    total_conversations: int
    total_sessions: int
    total_turns: int
    total_questions: int
    conversations_loaded: bool
    data_file: str

class ChatStorage:
    
    def __init__(self, data_url: str):
        self.data_url = data_url
        self.data_path = Path("/data/locomo10.json")
        self.conversations = []
        self.conversations_by_id = {}
        self.all_sessions = []
        self.all_questions = []
        self.is_loaded = False
    
    def download_data(self):
        if self.data_path.exists():
            print(f"Data file already exists at {self.data_path}")
            return True
        
        print(f"Downloading data from {self.data_url}")
        
        try:

            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            
            urllib.request.urlretrieve(self.data_url, self.data_path)
            print(f"Downloaded successfully to {self.data_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
        
    def load_from_json(self):

        if not self.data_path.exists():
            if not self.download_data():
                return False
        
        print(f"Loading conversations from {self.data_path}")
        
        if not self.data_path.exists():
            print(f"File not found: {self.data_path}")
            return False
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Expected JSON array, got {type(data)}")
                return False
            
            for conv_idx, conv_data in enumerate(data):
                try:                    

                    conversation_data = conv_data.get("conversation", {})
                    
                    speaker_a = conv_data.get("speaker_a") or conversation_data.get("speaker_a", "")
                    speaker_b = conv_data.get("speaker_b") or conversation_data.get("speaker_b", "")
                                        
                    sessions = {}
                    session_datetimes = {}
                    observations = {}
                    session_summaries = {}
                    event_summary = {}
                    
                    for key, value in conversation_data.items():

                        if key.startswith("session_"):

                            if key.endswith("_date_time") or key.endswith("_observation") or key.endswith("_summary"):
                                if key.endswith("_date_time"):
                                    session_datetimes[key] = value
                                elif key.endswith("_observation"):
                                    if isinstance(value, list):
                                        observations[key] = value
                                elif key.endswith("_summary") and not key.startswith("event"):
                                    session_summaries[key] = value

                            elif isinstance(value, list):
                                sessions[key] = [DialogTurn(**turn) for turn in value]
                        
                        elif key.startswith("events_session_"):

                            event_summary[key] = value
                        
                    if "observation" in conv_data:
                        obs_data = conv_data["observation"]
                        if isinstance(obs_data, dict):
                            observations.update(obs_data)
                    
                    if "session_summary" in conv_data:
                        sum_data = conv_data["session_summary"]
                        if isinstance(sum_data, dict):
                            session_summaries.update(sum_data)
                    
                    if "event_summary" in conv_data:
                        ev_data = conv_data["event_summary"]
                        if isinstance(ev_data, dict):
                            event_summary.update(ev_data)
                    
                    qa = None
                    if "qa" in conv_data and conv_data["qa"]:
                        qa = [Question(**q) for q in conv_data["qa"]]
                    
                    conversation = Conversation(
                        sample_id=conv_data.get("sample_id", ""),
                        speaker_a=speaker_a,
                        speaker_b=speaker_b,
                        sessions=sessions,
                        session_datetimes=session_datetimes,
                        observations=observations if observations else None,
                        session_summaries=session_summaries if session_summaries else None,
                        event_summary=event_summary if event_summary else None,
                        qa=qa
                    )
                    
                    self.conversations.append(conversation)
                    self.conversations_by_id[conversation.sample_id] = conversation
                                        
                    for session_key, turns in sessions.items():
                        session_text = " ".join([turn.text for turn in turns])
                        self.all_sessions.append({
                            "sample_id": conversation.sample_id,
                            "session_id": session_key,
                            "text": session_text,
                            "speakers": [conversation.speaker_a, conversation.speaker_b],
                            "num_turns": len(turns),
                            "date_time": session_datetimes.get(f"{session_key}_date_time"),
                            "turns": [turn.dict() for turn in turns]
                        })
                    
                    if qa:
                        for idx, question in enumerate(qa):
                            q_dict = {
                                "sample_id": conversation.sample_id,
                                "question_id": f"{conversation.sample_id}_q_{idx}",
                                "question": question.question,
                                "answer": question.get_answer(),
                                "category": question.category,
                                "evidence": question.evidence
                            }
                            self.all_questions.append(q_dict)
                    
                except Exception as e:
                    print(f"Error parsing conversation: {e}")
                    continue
            
            self.is_loaded = True
            print(f"Loaded {len(self.conversations)} conversations")
            print(f"Total sessions: {len(self.all_sessions)}")
            print(f"Total questions: {len(self.all_questions)}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_conversation(self, sample_id: str) -> Optional[Conversation]:
        return self.conversations_by_id.get(sample_id)
    
    def get_conversation_by_index(self, index: int) -> Optional[Conversation]:
        if 0 <= index < len(self.conversations):
            return self.conversations[index]
        return None
    
    def get_all_conversations(self, skip: int = 0, limit: int = 10) -> List[Conversation]:
        return self.conversations[skip:skip + limit]
    
    def get_conversation_session_by_id(self, sample_id: str, session_id: str) -> Optional[Dict]:
        conversation = self.get_conversation(sample_id)
        if not conversation:
            return None
        
        if session_id in conversation.sessions:
            return {
                "session_id": session_id,
                "turns": [turn.dict() for turn in conversation.sessions[session_id]],
                "date_time": conversation.session_datetimes.get(f"{session_id}_date_time"),
                "num_turns": len(conversation.sessions[session_id])
            }
        return None
    
    def get_conversation_session(self, sample_id: str, session_index: int) -> Optional[Dict]:
        conversation = self.get_conversation(sample_id)
        if not conversation:
            return None
        
        session_keys = sorted([k for k in conversation.sessions.keys() if k.startswith("session_")])
        
        if 0 <= session_index < len(session_keys):
            session_key = session_keys[session_index]
            return {
                "session_index": session_index,
                "session_id": session_key,
                "turns": [turn.dict() for turn in conversation.sessions[session_key]],
                "date_time": conversation.session_datetimes.get(f"{session_key}_date_time"),
                "num_turns": len(conversation.sessions[session_key])
            }
        return None
    
    def get_conversation_session_by_conv_index(self, conv_index: int, session_index: int) -> Optional[Dict]:
        conversation = self.get_conversation_by_index(conv_index)
        if not conversation:
            return None
        
        return self.get_conversation_session(conversation.sample_id, session_index)
    
    def get_conversation_questions(self, sample_id: str) -> List[Dict]:
        return [q for q in self.all_questions if q["sample_id"] == sample_id]
    
    def get_conversation_questions_by_index(self, conv_index: int) -> List[Dict]:
        conversation = self.get_conversation_by_index(conv_index)
        if not conversation:
            return []
        return self.get_conversation_questions(conversation.sample_id)
    
    def get_question_by_index(self, index: int) -> Optional[Dict]:
        if 0 <= index < len(self.all_questions):
            return self.all_questions[index]
        return None
    
    def get_sessions(self, sample_id: Optional[str] = None) -> List[Dict]:
        if sample_id:
            return [s for s in self.all_sessions if s["sample_id"] == sample_id]
        return self.all_sessions
    
    def get_questions(self, sample_id: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        questions = self.all_questions
        
        if sample_id:
            questions = [q for q in questions if q["sample_id"] == sample_id]
        
        if category:
            questions = [q for q in questions if q.get("category") == category]
        
        return questions
    
    def get_stats(self) -> Dict:
        total_turns = sum(s["num_turns"] for s in self.all_sessions)
        
        return {
            "total_conversations": len(self.conversations),
            "total_sessions": len(self.all_sessions),
            "total_turns": total_turns,
            "total_questions": len(self.all_questions),
            "conversations_loaded": self.is_loaded,
            "data_file": str(self.data_path)
        }

# Initialize storage
storage = ChatStorage(
    data_url=os.getenv("DATA_URL", "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json")
)

@app.on_event("startup")
async def startup_event():
    print("locomo starting...")
    
    storage.load_from_json()
    
    if storage.is_loaded:
        print(f"Ready with {len(storage.conversations)} conversations!")
    else:
        print("No data loaded. Check DATA_PATH environment variable.")

@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "locomo",
        "version": "1.0",
        "description": "Loads and serves LOCOMO conversation data",
        "data_format": "locomo10.json format with sessions, qa, observations, summaries",
        "endpoints": {
            "GET /conversations": "Get all conversations (paginated)",
            "GET /conversations/{sample_id}": "Get specific conversation",
            "GET /conversations/index/{index}": "Get conversation by index",
            "GET /conversations/{sample_id}/sessions/{session_id}": "Get specific session by session_id",
            "GET /conversations/index/{conv_index}/sessions/{session_index}": "Get session by conversation and session index",
            "GET /conversations/{sample_id}/questions": "Get questions for conversation",
            "GET /conversations/index/{index}/questions": "Get conversation by index and questions",
            "GET /stats": "Get statistics",
            "GET /health": "Health check"
        }
    }

@app.get("/health", tags=["Info"])
async def health():
    return {
        "status": "healthy" if storage.is_loaded else "not_loaded",
        "conversations_loaded": storage.is_loaded,
        "total_conversations": len(storage.conversations),
        "total_sessions": len(storage.all_sessions),
        "total_questions": len(storage.all_questions)
    }

@app.get("/stats", response_model=ConversationStats, tags=["Info"])
async def get_stats():
    stats = storage.get_stats()
    return ConversationStats(**stats)

@app.get("/conversations", tags=["Conversations"])
async def get_conversations(skip: int = 0, limit: int = 10):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    limit = min(limit, 10)
    conversations = storage.get_all_conversations(skip, limit)
    
    return {
        "total": len(storage.conversations),
        "skip": skip,
        "limit": limit,
        "count": len(conversations),
        "conversations": [c.dict() for c in conversations]
    }

@app.get("/conversations/{sample_id}", tags=["Conversations"])
async def get_conversation(sample_id: str):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    conversation = storage.get_conversation(sample_id)
    
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {sample_id} not found")
    
    return {
        "sample_id": conversation.sample_id,
        "speaker_a": conversation.speaker_a,
        "speaker_b": conversation.speaker_b,
        "sessions": conversation.sessions,
        "session_datetimes": conversation.session_datetimes
    }

@app.get("/conversations/index/{index}", tags=["Conversations"])
async def get_conversation_by_index(index: int):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    conversation = storage.get_conversation_by_index(index)
    
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation at index {index} not found")
    
    return {
        "sample_id": conversation.sample_id,
        "speaker_a": conversation.speaker_a,
        "speaker_b": conversation.speaker_b,
        "sessions": conversation.sessions,
        "session_datetimes": conversation.session_datetimes
    }

@app.get("/conversations/{sample_id}/sessions/{session_id}", tags=["Conversations"])
async def get_conversation_session_by_id(sample_id: str, session_id: str):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    session = storage.get_conversation_session_by_id(sample_id, session_id)
    
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found in conversation {sample_id}"
        )
    
    return session

@app.get("/conversations/index/{conv_index}/sessions/{session_index}", tags=["Conversations"])
async def get_conversation_session_by_index(conv_index: int, session_index: int):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    session = storage.get_conversation_session_by_conv_index(conv_index, session_index)
    
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_index} not found in conversation {conv_index}"
        )
    
    return session

@app.get("/conversations/{sample_id}/questions", tags=["Questions"])
async def get_questions_for_conversation(sample_id: str):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    questions = storage.get_questions(sample_id)
    
    return {
        "sample_id": sample_id,
        "total_questions": len(questions),
        "questions": questions
    }

@app.get("/conversations/index/{index}/questions", tags=["Questions"])
async def get_questions_by_conversation_index(index: int):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    conversation = storage.get_conversation_by_index(index)
    
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation at index {index} not found")
    
    questions = storage.get_questions(conversation.sample_id)
    
    return {
        "conversation_index": index,
        "sample_id": conversation.sample_id,
        "total_questions": len(questions),
        "questions": questions
    }