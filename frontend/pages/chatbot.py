import streamlit as st
from general_purpose_chat.general_purpose_chat_ui import render_general_purpose_chat
from plugins.Sidebar import render_save_chat

from mots_cle import ACRONYME

# 1. Configuration de la page (DOIT être le premier appel Streamlit)
st.set_page_config(page_title=f"Chatbot {ACRONYME}", page_icon="💧", layout="wide")

# 2. Rendu de l'interface de chat modulaire
# Cette fonction gère l'ensemble de l'interface utilisateur :
# - Sidebar avec contrôles de session et analyse Excel
# - Zone de chat principale avec historique
# - Traitement des fichiers et génération de graphiques
render_general_purpose_chat(title=f"Chatbot {ACRONYME}")

# 3. Composant de sauvegarde/restauration des conversations
# Permet aux utilisateurs d'exporter et importer leurs conversations
render_save_chat()
