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
from plugins.Chunks import render_chunk_card, render_chunks_panel
from plugins.Sidebar import render_sidebar
from plugins.Chat import render_chat


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Integrated", page_icon="🤖", layout="wide")
 
render_styles()
 
 
# ─── MAIN ─────────────────────────────────────────────────────────────────────
 
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    cfg = render_sidebar()
 
    st.title("🤖 RAG Intégré — Chatbot + Visualisation Chunks")

    render_chat(cfg)
 
if __name__ == "__main__":
    main()