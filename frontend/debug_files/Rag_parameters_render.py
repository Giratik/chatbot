"""
ui/sidebar.py
─────────────
Panneau latéral : sélection collection / modèle, filtres, paramètres RAG.
Retourne un dict de configuration consommé par render_chat().
"""
import json
import streamlit as st
from plugins import APIclient as api






def render_sidebar() -> dict:
    """Rend la sidebar et retourne la config sélectionnée."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration RAG")

        try:
            collections = api.list_collections()
            models = api.list_generative_models()
        except Exception as e:
            st.error(f"Erreur d'initialisation : {e}")
            st.stop()

        api_url = api.BASE_URL
        st.markdown(
            f"<span class='badge badge-ok'>● {api_url}</span>",
            unsafe_allow_html=True,
        )

        if not collections:
            st.warning("Aucune collection trouvée.")
            st.stop()
        if not models:
            st.warning("Aucun modèle génératif trouvé.")
            st.stop()

        selected_collection = st.selectbox("Collection ChromaDB", collections)
        selected_model = st.selectbox("Modèle LLM Ollama", models)

        doc_dates = api.list_doc_dates(selected_collection)

        st.markdown("---")
        selected_doc_date = st.selectbox(
            "Filtrer par date du document",
            ["Toutes"] + doc_dates,
            help="Si une date est sélectionnée, seuls les chunks issus de documents de cette date seront recherchés.",
        )
        selected_doc_date = "" if selected_doc_date == "Toutes" else selected_doc_date

        st.markdown("---")
        n_results = st.slider("Chunks à injecter", 1, 500, 250)
        seuil = st.slider("Seuil de distance (cosine)", 0.1, 1.0, 0.7, 0.05)

        st.markdown("---")
        st.markdown("**🔬 Stratégie de recherche**")
        use_hyde = st.toggle("HyDE (réponse hypothétique)", value=True)
        use_expansion = st.toggle("Query expansion (synonymes)", value=True)
        alpha = st.slider("Vectoriel ← → BM25", 0.0, 1.0, 0.5, 0.05)

        st.markdown("---")
        st.markdown("**🎯 Reranking**")
        use_reranker = st.toggle("Reranker (bge-reranker-v2-gemma)", value=False)

        st.markdown("---")
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

    return {
        "collection": selected_collection,
        "model": selected_model,
        "doc_date_filter": selected_doc_date,
        "n_results": n_results,
        "seuil": seuil,
        "use_hyde": use_hyde,
        "use_expansion": use_expansion,
        "alpha": alpha,
        "use_reranker": use_reranker,
    }
