"""
ui/sidebar.py
─────────────
Panneau latéral : sélection collection / modèle, filtres, paramètres RAG.
Retourne un dict de configuration consommé par render_chat().
"""
import json
import streamlit as st
from plugins import APIclient as api

def render_save_chat():
    with st.sidebar:
        st.info("L'erreur KeyError: 'display_content' est normale lors d'un changement de chat à un autre, les deux fenêtres de chats n'ont pas un affichage compatible. Il faut recharger la page/commencer une nouvelle conversation.")
        # ==========================================
            # NOUVEAU : Sauvegarde et Chargement (JSON)
            # =========================================
        st.markdown("**💾 Sauvegarde & Historique**")

        # 1. EXPORT : Bouton pour télécharger la conversation
        if "messages" in st.session_state and st.session_state.messages:
            # On convertit la liste de messages en chaîne JSON
            chat_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
            
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
                    # On lit et on décode le contenu du fichier
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    loaded_messages = json.loads(file_content)
                    
                    # On met à jour l'état de la session
                    st.session_state.messages = loaded_messages
                    st.success("Conversation restaurée !")
                    st.rerun()  # Recharge la page pour afficher les messages
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier : {e}")
    
    
