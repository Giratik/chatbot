
# frontend/plugins/Sidebar.py - Composant de Sidebar et Sauvegarde

import json
import streamlit as st
from plugins import wrapper_API as api

from utility.session_state_central import SK, get, set as ss_set

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