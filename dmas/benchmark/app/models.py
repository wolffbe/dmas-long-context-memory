from pydantic import BaseModel
from typing import List, Dict, Optional, Any


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