# frontend/main.py


import streamlit as st
import os

LOGO_PATH = "ressource/Eau_de_Paris_bleu.svg.png"
IS_DEV = os.environ.get("IS_DEV", "no")

st.set_page_config(page_title="Chatbot EDP-IA", page_icon=":material/robot:", layout="wide")
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH)
def main():
    if "is_dev" not in st.session_state:
        st.session_state.is_dev = IS_DEV

    #if "rag_config" not in st.session_state:
    #    st.session_state.rag_config = set_rag_stats()

    # Déclaration des pages
    page_chat = st.Page("pages/chatbot.py", title="Chatbot", icon=":material/chat:", default=True)
    page_changelog = st.Page("pages/Changelog.py", title="Changelog", icon=":material/description:")
    page_tool_calling = st.Page("debug_files/off_tool_calling.py", title="tool_calling")

    # Construction dynamique de la navigation
    pages_visibles = [page_chat]
    pages_visibles.append(page_changelog)

    # Ajout conditionnel de la page de config
    if st.session_state.is_dev == "yes":
        pages_visibles.append(page_tool_calling)

    # Exécution de la navigation
    pg = st.navigation(pages_visibles)
    pg.run()

if __name__ == "__main__":
    main()