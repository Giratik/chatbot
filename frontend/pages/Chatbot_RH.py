"""
Chatbot_RH.py — Interface Streamlit RAG (frontend uniquement)
──────────────────────────────────────────────────────
Toute la logique métier est dans backend.py.
Tous les composants UI sont dans le dossier ui/.

Lancement :
    streamlit run app.py
"""


import streamlit as st
from plugins.Styles import render_styles
from plugins.Sidebar import render_save_chat
from plugins.Chat import render_rag_chat


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Integrated", page_icon="🤖", layout="wide")
 
render_styles()
 
 
# ─── MAIN ─────────────────────────────────────────────────────────────────────
 
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #st.sidebar.info("S'il y a des problèmes pour récupérer les fichiers, c'est un problème de pare-feu")
    cfg = {
    "collection": "dummy_rh",
    "model": "gemma4:e4b",
    "doc_date_filter": "",
    "n_results": 250,
    "seuil": 0.6,
    "use_hyde": True,
    "use_expansion": True,
    "alpha": 0.5,
    "use_reranker": False
    }

    #st.json (cfg)
 
    st.title("Chatbot spécialisé question RH")

    render_rag_chat(cfg)

    render_save_chat()  # Rendre le composant de sauvegarde du chat
 
if __name__ == "__main__":
    main()