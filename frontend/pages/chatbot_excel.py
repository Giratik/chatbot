import streamlit as st
from renders.chat_ui_v2 import render_chat_v2
from plugins.Sidebar import render_save_chat

# 1. Configuration de la page (DOIT être le premier appel Streamlit)
st.set_page_config(page_title="Chatbot EDP", page_icon="💧", layout="wide")


# 2. Rendu de l'interface de chat modulaire
render_chat_v2(title="Chatbot EDP")

render_save_chat()  # Rendre le composant de sauvegarde du chat