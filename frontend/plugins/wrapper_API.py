"""
plugins/wrapper_API.py
────────────────
Wrapper HTTP vers l'API FastAPI RAG.
Chaque fonction reproduit la signature attendue par les modules ui/
afin de rester un drop-in replacement des appels directs à backend.py.
"""

from __future__ import annotations
import re

import requests
from typing import Generator, List, Dict, Any, Optional

# ── URL de base (peut être surchargée via st.secrets ou variable d'env) ───────
import os
BASE_URL = os.getenv("API_URL", os.getenv("RAG_API_URL", "http://backend:8000"))

# ─── helpers ──────────────────────────────────────────────────────────────────
 
def _get(path: str, **kwargs) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", **kwargs)
    resp.raise_for_status()
    return resp.json()
 
 
def _post(path: str, payload: dict, **kwargs) -> Any:
    resp = requests.post(f"{BASE_URL}{path}", json=payload, **kwargs)
    resp.raise_for_status()
    return resp.json()
 
 

# ─── Recherche hybride ────────────────────────────────────────────────────────
 
def retrieve_context_hybrid(
    collection_name: str,
    query: str,
    model: str,
    n_results: int = 5,
    seuil: float = 0.5,
    alpha: float = 0.5,
    use_hyde: bool = False,
    use_expansion: bool = False,
    doc_date_filter: str = "",
) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    data = _post("/rag/search", {
        "collection_name": collection_name,
        "query": query,
        "model": model,
        "n_results": n_results,
        "seuil": seuil,
        "alpha": alpha,
        "use_hyde": use_hyde,
        "use_expansion": use_expansion,
        "doc_date_filter": doc_date_filter,
    })
    return data["contexts"], data["sources"], data["detailed_chunks"]


def get_registry_evolve():
   try:
       registry_entries = _get("/rag/registry_evolve")
       return registry_entries # => bonne réponse
   except Exception as e:
       print(f"Erreur registry : {e}")
       return []