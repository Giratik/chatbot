"""
Main.py - Point d'entrée principal du chatbot généraliste
───────────────────────────────────────────────────────────────
Rôle : Interface principale pour le traitement de fichiers et demandes générales
- Traite les questions générales
- Analyse les fichiers Excel, Word, PDF, etc.
- Génère des graphiques et exécute des requêtes SQL
- Gestion unifiée de session

Architecture :
- Utilise general_purpose_chat_ui.py pour l'interface principale
- Intègre Sidebar.py pour la sauvegarde des conversations
- Ne dépend pas des composants RAG (contrairement à Chatbot_RH.py)

Différence avec Chatbot_RH.py :
- Ce fichier est pour les demandes générales et l'analyse de fichiers
- Chatbot_RH.py est spécialisé pour les questions RH avec accès RAG
"""

import streamlit as st
from plugins.general_purpose_chat_ui import render_general_purpose_chat
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