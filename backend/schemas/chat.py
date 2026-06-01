from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    think: bool = False

class RetrieveRequest(BaseModel):
    collection_name: str
    query: str
    model: str
    n_results: int = 5
    seuil: float = 0.5
    alpha: float = 0.5
    use_hyde: bool = False
    use_expansion: bool = False
    use_reranker: bool = True
    doc_date_filter: str = ""

class RewriteRequest(BaseModel):
    query: str
    model: str
    chat_history: List[Message] = []

class StreamChatRequest(BaseModel):
    collection_name: str
    query: str
    model: str
    system_prompt_context: str
    chat_history: List[Message] = []