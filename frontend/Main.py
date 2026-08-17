# frontend/main.py


import streamlit as st
import os

IS_DEV = os.environ.get("IS_DEV", "no")

def main():
    if "is_dev" not in st.session_state:
        st.session_state.is_dev = IS_DEV

    #if "rag_config" not in st.session_state:
    #    st.session_state.rag_config = set_rag_stats()

    # Déclaration des pages
    page_chat = st.Page("pages/chatbot.py", title="Chatbot", icon="💬", default=True)
    page_changelog = st.Page("pages/Changelog.py", title="Changelog", icon="📝")
    #page_debug = st.Page("debug_files/Rag_parameters_render.py", title="Configuration", icon="⚙️")
    page_tool_calling = st.Page("debug_files/off_tool_calling.py", title="tool_calling")

    # Construction dynamique de la navigation
    pages_visibles = [page_chat]

    # Ajout conditionnel de la page de config
    if st.session_state.is_dev == "yes":
        pages_visibles.append(page_changelog)
        #pages_visibles.append(page_debug)
        pages_visibles.append(page_tool_calling)

    # Exécution de la navigation
    pg = st.navigation(pages_visibles)
    pg.run()

if __name__ == "__main__":
    main()