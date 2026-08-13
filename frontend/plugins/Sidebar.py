
# frontend/plugins/Sidebar.py - Composant de Sidebar et Sauvegarde

import json
import streamlit as st
from plugins import wrapper_API as api

from utility.session_state_central import SK, get, set as ss_set

def render_sidebar() -> dict:
    """
    Rend la sidebar et retourne la configuration sélectionnée.
    Cette fonction n'est actuellement pas utilisée dans Main.py mais est disponible
    pour une configuration RAG avancée si nécessaire.

    Returns:
        dict: Configuration RAG avec les paramètres sélectionnés par l'utilisateur
    """
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



def render_save_chat():
    """
    Composant de sauvegarde et restauration des conversations.
    Cette fonction est utilisée par Main.py et Chatbot_RH.py pour permettre
    aux utilisateurs d'exporter et importer leurs conversations.

    Fonctionnalités :
    - Export de la conversation au format JSON
    - Import et restauration de conversations sauvegardées
    - Gestion des erreurs de format
    """
    with st.sidebar:

        # Sauvegarde et Chargement (JSON)
        # =========================================
        st.markdown("**💾 Sauvegarde & Historique**")

        # 1. EXPORT : Bouton pour télécharger la conversation
        messages = get(SK.MESSAGES)                          # ✅
        if messages:
            chat_json = json.dumps(messages, ensure_ascii=False, indent=2)

            st.download_button(
                label="📥 Exporter la conversation",
                data=chat_json,
                file_name="historique_conversation.json",
                mime="application/json",
                help="Télécharge l'historique actuel au format JSON pour le reprendre plus tard."
            )

        # 2. IMPORT : Uploader pour charger un fichier JSON
        uploaded_file = st.file_uploader("📂 Reprendre une conversation", type=["json"])

        if uploaded_file is not None:
            if st.button("Restaurer cette conversation"):
                try:
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    loaded_messages = json.loads(file_content)

                    ss_set(SK.MESSAGES, loaded_messages)     # ✅
                    st.success("Conversation restaurée !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier : {e}")