"""
ui/chat.py
──────────
Colonne de conversation : historique, saisie, pipeline RAG complet,
streaming de la réponse via l'API FastAPI.
"""

import streamlit as st
from plugins import APIclient as api


def render_chat(cfg: dict) -> None:
    """Affiche la colonne de chat et exécute le pipeline RAG à chaque message."""
    st.markdown("### 💬 Conversation")

    # ── Historique affiché ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not (prompt := st.chat_input("Posez une question sur vos documents...")):
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── 1. Réécriture contextuelle ────────────────────────────────────────
        history_for_rewrite = st.session_state.messages[:-1]
        try:
            standalone_query = api.rewrite_query(
                query=prompt,
                model=cfg["model"],
                chat_history=history_for_rewrite,
            )
        except Exception as e:
            st.error(f"Erreur lors de la réécriture : {e}")
            return

        if standalone_query.lower().strip() != prompt.lower().strip():
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
        system_prompt = api.build_system_prompt(context_str)

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

        # ── 4. Sauvegarde ─────────────────────────────────────────────────────
        st.session_state.messages.append({"role": "assistant", "content": full_response})