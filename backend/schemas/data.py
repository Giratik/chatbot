from pydantic import BaseModel
from typing import List, Dict, Any

class ChatRequest_csv(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    session_id: str = "default"
    mode: str = "discussion"
    think: bool = False

class SqlRequest(BaseModel):
    sql: str
    session_id: str = "default"

class SessionRequest(BaseModel):
    session_id: str = "default"