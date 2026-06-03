# frontend/Main.py
import streamlit as st
from renders.chat_ui import render_chat

# 1. Configuration de la page (DOIT être le premier appel Streamlit)
st.set_page_config(page_title="Chatbot EDP", page_icon="💧", layout="wide")

# 2. Rendu de l'interface de chat modulaire
render_chat(title="Chatbot EDP")