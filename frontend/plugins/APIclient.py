"""
plugins/APIclient.py
────────────────
Wrapper HTTP vers l'API FastAPI RAG.
Chaque fonction reproduit la signature attendue par les modules ui/
afin de rester un drop-in replacement des appels directs à backend.py.
"""

from __future__ import annotations

import requests
from typing import Generator, List, Dict, Any, Optional

# ── URL de base (peut être surchargée via st.secrets ou variable d'env) ───────
import os
BASE_URL = os.getenv("API_URL", os.getenv("RAG_API_URL", "http://localhost:8000"))


# ─── helpers ──────────────────────────────────────────────────────────────────

def _get(path: str, **kwargs) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", **kwargs)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: dict, **kwargs) -> Any:
    resp = requests.post(f"{BASE_URL}{path}", json=payload, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ─── Collections & modèles ────────────────────────────────────────────────────

def list_collections() -> List[str]:
    """Retourne la liste des collections ChromaDB disponibles."""
    return _get("/rag/collections")["collections"]


def list_generative_models() -> List[str]:
    """Retourne la liste des modèles Ollama génératifs disponibles."""
    return _get("/rag/models")["models"]


# ─── Dates disponibles ────────────────────────────────────────────────────────

def list_doc_dates(collection_name: str) -> List[str]:
    """Retourne les dates de documents disponibles pour une collection."""
    try:
        return _get(f"/rag/collections/{collection_name}/dates")
    except Exception:
        # endpoint optionnel : on renvoie une liste vide si absent
        return []


# ─── Réécriture de requête ────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    model: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """Réécrit la requête utilisateur en tenant compte de l'historique."""
    data = _post("/rag/rewrite", {
        "query": query,
        "model": model,
        "chat_history": chat_history,
    })
    return data["rewritten_query"]


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
    use_reranker: bool = True,
    doc_date_filter: str = "",
) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Lance la recherche hybride et retourne (contexts, sources, detailed_chunks)."""
    data = _post("/rag/search", {
        "collection_name": collection_name,
        "query": query,
        "model": model,
        "n_results": n_results,
        "seuil": seuil,
        "alpha": alpha,
        "use_hyde": use_hyde,
        "use_expansion": use_expansion,
        "use_reranker": use_reranker,
        "doc_date_filter": doc_date_filter,
    })
    return data["contexts"], data["sources"], data["detailed_chunks"]


# ─── Streaming de la réponse ──────────────────────────────────────────────────

def stream_answer(
    system_prompt: str,
    query: str,
    model: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[str, None, None]:
    """Génère la réponse token par token via l'endpoint /rag/stream_answer."""
    payload = {
        "system_prompt": system_prompt,
        "query": query,
        "model": model,
        "chat_history": chat_history or [],
    }
    with requests.post(
        f"{BASE_URL}/rag/stream_answer",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


# ─── Prompt système ───────────────────────────────────────────────────────────

def build_system_prompt(context_str: str) -> str:
    """Construit le prompt système à partir du contexte récupéré.

    Cette fonction est conservée côté client pour éviter un aller-retour
    réseau inutile : le contexte est déjà dans le frontend après /rag/search.
    """
    return (
        "Tu es un assistant expert. Réponds uniquement en te basant sur le contexte suivant.\n\n"
        f"Contexte :\n{context_str}\n\n"
        "Si la réponse n'est pas dans le contexte, dis-le clairement."
    )