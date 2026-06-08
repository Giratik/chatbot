import streamlit as st
from plugins.general_purpose_chat_ui import render_general_purpose_chat
from plugins.Sidebar import render_save_chat

# 1. Configuration de la page (DOIT être le premier appel Streamlit)
st.set_page_config(page_title="Chatbot EDP", page_icon="💧", layout="wide")


# 2. Rendu de l'interface de chat modulaire
render_general_purpose_chat(title="Chatbot EDP")

render_save_chat()  # Rendre le composant de sauvegarde du chat