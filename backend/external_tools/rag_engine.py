#backend/engines/rag_engine.py

import re
import httpx
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import VectorParams, Distance, PayloadSchemaType
from qdrant_client.http.models import PointStruct
import uuid
from datetime import datetime, timezone

from services.ollama_client import inferring_ollama
from core.config import QDRANT_HOST, QDRANT_PORT, OLLAMA_HOST, EMBEDDING_MODEL, CONTEXT_SIZE
from typing import Any


# ─── CLIENTS ──────────────────────────────────────────────────────────────────

def make_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=int(QDRANT_PORT))

from services.ollama_client import client as ollama_sdk_client

def make_ollama_client():
    return ollama_sdk_client  # le vrai Client SDK Ollama

def embed(texts: list[str], ollama_host: str, model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Appel direct à l'API Ollama pour produire les embeddings."""
    vectors = []
    with httpx.Client(timeout=60) as client:
        for text in texts:
            resp = client.post(
                f"{ollama_host}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
    return vectors


#def list_collections(qdrant_client: QdrantClient) -> list[str]:
#    return sorted(c.name for c in qdrant_client.get_collections().collections)

# ─── REGISTRY DES COLLECTIONS ────────────────────────────────────────────────
REGISTRY_COLLECTION = "_registry"

def ensure_registry(qdrant_client: QdrantClient) -> None:
    """Crée la collection _registry si elle n'existe pas."""
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if REGISTRY_COLLECTION not in existing:
        qdrant_client.create_collection(
            collection_name=REGISTRY_COLLECTION,
            # Vecteurs factices de dimension 1 — le registry n'est pas requêté
            # par similarité, uniquement par scroll/filtre.
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )
        qdrant_client.create_payload_index(
            collection_name=REGISTRY_COLLECTION,
            field_name="collection_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )

def list_registry(qdrant_client: QdrantClient) -> list[dict]:
    """Retourne toutes les entrées du registry triées par nom de collection."""
    ensure_registry(qdrant_client)
    records = []
    offset = None
    while True:
        batch, offset = qdrant_client.scroll(
            collection_name=REGISTRY_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(batch)
        if offset is None:
            break
    return sorted(
        [r.payload for r in records if r.payload],
        key=lambda x: x.get("collection_name", ""),
    )


def registry_for_tool_calling(client, role: str = "") -> list[dict]: #accès basé sur les rôles (RBAC)
    entries = list_registry(client)
    result = []
    for e in entries:
        if not e.get("active", True):
            continue
        allowed = e.get("allowed_roles", [])
        # Accessible si : pas de restriction, ou le rôle est dans la liste
        if not allowed or role in allowed or role == "admin":
            result.append({"nom": e["collection_name"], "description": e["description"]})
    return result

def list_doc_dates(qdrant_client: QdrantClient, collection_name: str) -> list[str]:
    """Parcourt la collection Qdrant pour extraire les dates uniques."""
    dates = set()
    offset = None
    while True:
        records, offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            if r.payload and r.payload.get("doc_date"):
                dates.add(r.payload["doc_date"])
        if offset is None:
            break
    return sorted(list(dates))




# ─── QUERY AUGMENTATION ───────────────────────────────────────────────────────

def expand_query(ollama_client: Any, model: str, query: str) -> list[str]:
    prompt = (
        "Reformule cette question en 3 variantes courtes avec des synonymes différents.\n"
        "Retourne UNIQUEMENT les 3 reformulations, une par ligne, sans numérotation ni explication.\n"
        f"Question : {query}"
    )
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4},
        )
        variants = [l.strip() for l in resp["message"]["content"].split("\n") if l.strip()]
        return [query] + variants[:3]
    except Exception:
        return [query]


def hyde_query(ollama_client: Any, model: str, query: str) -> str:
    prompt = (
        "Rédige un court paragraphe (3-4 phrases) qui serait une réponse plausible à cette question.\n"
        "Utilise un vocabulaire précis et varié. N'indique pas que c'est hypothétique.\n"
        f"Question : {query}"
    )
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        return resp["message"]["content"].strip()
    except Exception:
        return query


# ─── TOKENISATION ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ─── HYBRID SEARCH ────────────────────────────────────────────────────────────

def retrieve_context_hybrid(
    qdrant_client: QdrantClient,
    collection_name: str,
    query: str,
    ollama_client: Any,
    model: str,
    n_results: int,
    seuil: float,
    alpha: float,
    use_hyde: bool,
    use_expansion: bool,
    doc_date_filter: str = "",
) -> tuple[list[str], list[tuple], list[dict]]:
    
    queries = [query]
    if use_expansion:
        queries = expand_query(ollama_client, model, query)
    if use_hyde:
        queries.append(hyde_query(ollama_client, model, query))

    per_query = max(5, n_results // len(queries))

    # ── Récupération vectorielle Qdrant ───────────────────────────────────────
    candidates: dict[str, dict] = {}
    
    qdrant_filter = None
    if doc_date_filter:
        qdrant_filter = Filter(must=[FieldCondition(key="doc_date", match=MatchValue(value=doc_date_filter))])

    for q in queries:
        try:
            q_vector = embed([q], OLLAMA_HOST, EMBEDDING_MODEL)[0]
            result = qdrant_client.query_points(
                collection_name=collection_name,
                query=q_vector,
                query_filter=qdrant_filter,
                limit=per_query,
                with_payload=True
            )
            
            for hit in result.points:
                dist = max(0.0, 1.0 - hit.score)
                if dist <= seuil and hit.id not in candidates:
                    candidates[hit.id] = {
                        "document": hit.payload.get("document", ""),
                        "metadata": hit.payload,
                        "vecto_distance": dist,
                    }
        except Exception as e:
            import logging
            logging.warning(f"Erreur sur la query '{q}': {e}")
            continue

    if not candidates:
        return [], [], []

    ids = list(candidates.keys())
    docs = [candidates[i]["document"] for i in ids]
    metas = [candidates[i]["metadata"] for i in ids]
    vecto_distances = [candidates[i]["vecto_distance"] for i in ids]

    # ── Scores normalisés ─────────────────────────────────────────────────────
    vecto_scores = [1 - d / 2 for d in vecto_distances]
    max_v = max(vecto_scores) or 1
    vecto_scores_norm = [s / max_v for s in vecto_scores]

    corpus_tokens = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(tokenize(query))
    max_b = max(bm25_scores) or 1
    bm25_scores_norm = [s / max_b for s in bm25_scores]

    hybrid_scores = [
        alpha * vecto_scores_norm[i] + (1 - alpha) * bm25_scores_norm[i]
        for i in range(len(ids))
    ]

    # ── Hybrid ranking initial ────────────────────────────────────────────────
    ranked = sorted(
        zip(hybrid_scores, vecto_distances, bm25_scores, docs, metas),
        key=lambda x: x[0],
        reverse=True,
    )[:n_results]

    ranked_with_rerank = [(*item, 0.0) for item in ranked]

    # ── Construction des résultats ────────────────────────────────────────────
    contexts: list[str] = []
    sources: list[tuple] = []
    detailed_chunks: list[dict] = []
    seen_sources: set[str] = set()

    for hybrid_score, vecto_dist, bm25_score, doc, meta, rerank_score in ranked_with_rerank:
        if "source" in meta and "page" in meta:
            source_name = f"📄 {meta['source']} (Page {meta['page']})"
            source_url = meta.get("source_url", "").strip()
            if source_url:
                source_name += f" — [Ouvrir le lien]({source_url})"
            chunk_type = "pdf"
            doc_date = meta.get("doc_date", "")
        elif "acronyme" in meta:
            source_name = f"📚 Lexique : {meta['acronyme']}"
            chunk_type = "lexique"
            doc_date = ""
        else:
            source_name = "Document inconnu"
            chunk_type = "unknown"
            doc_date = ""

        context_line = f"Extrait de {source_name}"
        if doc_date:
            context_line += f" [Document du {doc_date}]"
        context_line += f" :\n{doc}"
        contexts.append(context_line)

        if source_name not in seen_sources:
            sources.append((source_name, hybrid_score, vecto_dist, doc_date))
            seen_sources.add(source_name)

        detailed_chunks.append({
            "source": source_name,
            "type": chunk_type,
            "document": doc,
            "metadata": meta,
            "hybrid_score": hybrid_score,
            "vecto_distance": vecto_dist,
            "bm25_score": bm25_score,
            "doc_date": doc_date,
            "rerank_score": rerank_score,
            "source_url": meta.get("source_url", ""),
        })

    return contexts, sources, detailed_chunks


# ─── GÉNÉRATION LLM ───────────────────────────────────────────────────────────
from core.config import SYSTEM_PROMPT

def build_system_prompt(context_str: str) -> str:
    return f""" {SYSTEM_PROMPT} 
    CONTEXTE :{context_str} """


def rewrite_query(
    ollama_client,
    model: str,
    query: str,
    chat_history: list[dict],
) -> str:
    if not chat_history:
        return query
 
    MAX_TURNS = 4
    recent = chat_history[-(MAX_TURNS * 2):]
    history_str = "\n".join(
        f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'} : {m['content']}"
        for m in recent
    )
 
    prompt = (
        "Tu es un assistant qui reformule des questions.\n"
        "Voici l'historique récent de la conversation :\n"
        f"{history_str}\n\n"
        "Nouvelle question de l'utilisateur : « {query} »\n\n"
        "Ta tâche : si cette question contient des pronoms, ellipses ou références "
        "implicites à l'historique (ex: 'y a t il un délai ?', 'et pour lui ?', "
        "'quel est ce montant ?'), reformule-la en incluant explicitement "
        "le sujet principal de la conversation et toute population ou cas particulier "
        "mentionné dans l'historique.\n"
        "Si la question est déjà autonome, retourne-la EXACTEMENT telle quelle.\n"
        "Retourne UNIQUEMENT la question reformulée, sans explication ni ponctuation "
        "supplémentaire."
    ).format(query=query)
 
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        rewritten = resp["message"]["content"].strip().strip("«»\"'")
        return rewritten if len(rewritten) > 5 else query
    except Exception:
        return query
 