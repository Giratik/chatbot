"""
ui/chat.py
──────────
Colonne de conversation : historique, saisie, pipeline RAG complet,
streaming de la réponse via l'API FastAPI.
"""

import re
import streamlit as st
from plugins import APIclient as api


def _render_sources(citations: list[str]) -> None:
    """Affiche les sources citées dans un expander sous la réponse."""
    if not citations:
        return
    with st.expander(f"📚 Sources citées ({len(citations)})", expanded=False):
        for src in citations:
            st.markdown(f"- 📄 {src}")
 
 
def render_rag_chat(cfg: dict) -> None:
    """Affiche la colonne de chat et exécute le pipeline RAG à chaque message."""
    #st.markdown("### 💬 Conversation")
 
    # ── Historique affiché ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_sources(msg["citations"])
 
    if not (prompt := st.chat_input("Posez une question sur vos documents...")):
        return
 
    st.session_state.messages.append({"role": "user", "content": prompt})
 
    with st.chat_message("user"):
        st.markdown(prompt)
 
    with st.chat_message("assistant"):
 
        # ── 1. Réécriture contextuelle ────────────────────────────────────────
        history_for_rewrite = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        try:
            standalone_query = api.rewrite_query(
                query=prompt,
                model=cfg["model"],
                chat_history=history_for_rewrite,
            )
        except Exception as e:
            st.error(f"Erreur lors de la réécriture : {e}")
            return
 
        # Afficher le badge seulement si la reformulation apporte un vrai changement
        # (on ignore la ponctuation, la casse et les espaces superflus)
        def _normalize(s: str) -> str:
            return re.sub(r"[^\w\s]", "", s.lower()).split()
 
        if _normalize(standalone_query) != _normalize(prompt):
            st.markdown(
                f"<div class='rewrite-badge'>🔄 Query : {standalone_query}</div>",
                unsafe_allow_html=True,
            )
 
        # ── 2. Recherche hybride ──────────────────────────────────────────────
        with st.status("🔍 Recherche dans les documents...", expanded=True) as status:
            try:
                contexts, sources, detailed_chunks = api.retrieve_context_hybrid(
                    collection_name=cfg["collection"],
                    query=standalone_query,
                    model=cfg["model"],
                    n_results=cfg["n_results"],
                    seuil=cfg["seuil"],
                    alpha=cfg["alpha"],
                    use_hyde=cfg["use_hyde"],
                    use_expansion=cfg["use_expansion"],
                    use_reranker=cfg["use_reranker"],
                    doc_date_filter=cfg.get("doc_date_filter", ""),
                )
            except Exception as e:
                status.update(label=f"Erreur de recherche : {e}", state="error")
                return
 
            if not contexts:
                status.update(label="Aucun document pertinent trouvé.", state="error")
                context_str = "Aucun contexte pertinent trouvé."
            else:
                nb_queries = 1 + (3 if cfg["use_expansion"] else 0) + (1 if cfg["use_hyde"] else 0)
                status.update(
                    label=f"{len(contexts)} extraits (sur {nb_queries} requêtes)",
                    state="complete",
                )
                context_str = "\n\n---\n\n".join(contexts)
 
            st.session_state.last_chunks = detailed_chunks
 
        # ── 3. Génération streamée ────────────────────────────────────────────
        placeholder = st.empty()
        full_response = ""
        # On passe les chunks pour étiqueter le contexte par source
        system_prompt = api.build_system_prompt(context_str, detailed_chunks)
 
        try:
            for token in api.stream_answer(
                system_prompt=system_prompt,
                query=prompt,
                model=cfg["model"],
                chat_history=history_for_rewrite,
            ):
                if token.startswith("ERROR:"):
                    st.error(token[6:])
                    return
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
 
        except Exception as e:
            st.error(f"Erreur lors de la génération : {e}")
            return
 
        # ── 4. Parse et affiche les sources citées ────────────────────────────
        clean_response, citations = api.extract_citations(full_response)
 
        # Remplacer la réponse brute par la version sans balises dans le chat
        if citations:
            placeholder.markdown(clean_response)
 
        _render_sources(citations)
 
        # ── 5. Sauvegarde ─────────────────────────────────────────────────────
        st.session_state.messages.append({
            "role": "assistant",
            "content": clean_response,
            "citations": citations,
        })
 