# frontend/pages/1_⚙️_Configuration.py

import streamlit as st
import datetime
from plugins.APIclient import list_collections

st.set_page_config(page_title="Configuration RAG", page_icon="⚙️", layout="wide")

# ─── FONCTION D'INITIALISATION ────────────────────────────────────────────────
def init_session_state():
    """Initialise les variables de session si elles n'existent pas encore."""
    
    # 1. Date automatique pour le prompt
    if "system_prompt" not in st.session_state:
        mois_fr = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
                   "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        now = datetime.datetime.now()
        date_actuelle = f"{mois_fr[now.month - 1]} {now.year}"
        
        st.session_state.system_prompt = f"""Tu es un assistant IA expert, concis et professionnel.
Ta mission est de répondre à la question de l'utilisateur en utilisant UNIQUEMENT le contexte fourni ci-dessous.
Si la réponse n'est pas dans le contexte, dis poliment "Je ne trouve pas cette information dans les documents fournis", et n'invente rien.
Réponds en français.

RÈGLES IMPORTANTES :
- Nous sommes en {date_actuelle}.
- Les dates des documents sont indiquées entre crochets [Document du YYYY-MM-DD].
- Si plusieurs documents traitent le même sujet avec des dates différentes, PRIORISE TOUJOURS le document le plus récent et considère les autres comme caduques."""



import re

def set_rag_stats():
    # J'assume que list_collections() est appelée correctement ici, 
    # potentiellement en lui passant ton client ChromaDB si nécessaire.
    collections_disponibles = list_collections()
    
    # Sécurité : au cas où ChromaDB est vide ou inaccessible
    if not collections_disponibles:
        s = "aucune_collection"
    else:
        # Recherche regex : on cherche "RH" (insensible à la casse grâce à re.IGNORECASE)
        # c'est-à-dire que ça matchera "rh", "RH", "documents_RH", "Rh_test", etc.
        collection_rh = next(
            (c for c in collections_disponibles if re.search(r'test', c, re.IGNORECASE)), 
            None # Valeur par défaut si rien n'est trouvé
        )
        
        # Si la regex trouve une correspondance, on l'utilise.
        # Sinon (fallback), on prend le premier élément de la liste pour ne pas faire planter l'app.
        s = collection_rh if collection_rh else collections_disponibles[0]

    return {
        "collection": s,
        "model": "gemma4:e4b",
        "doc_date_filter": "",
        "n_results": 250,
        "seuil": 0.6,
        "use_hyde": True,
        "use_expansion": True,
        "alpha": 0.5,
    }


# ─── RENDU DE LA PAGE ─────────────────────────────────────────────────────────
def render_config_page():
    init_session_state()
    
    st.title("⚙️ Configuration de la Session")
    st.info("Les modifications effectuées ici s'appliquent uniquement à votre session en cours et seront réinitialisées au rechargement complet de l'application. Les paramètres de bases sont déja optimaux pour la récupération d'information")

    # -- Section 1 : Paramètres du RAG --
    st.header("Paramètres du RAG (cfg)")
    col1, col2 = st.columns(2)
    
    with col1:
        # --- NOUVEAU BLOC DYNAMIQUE POUR LES COLLECTIONS ---
        try:
            # 1. Récupération des collections via l'API frontend
            collections_disponibles = list_collections()

            # Si la BDD est vide, on fournit un fallback
            if not collections_disponibles:
                collections_disponibles = ["aucune_collection_trouvee"]
                st.warning("Aucune collection trouvée dans ChromaDB.")

        except Exception as e:
            # Sécurité : Si l'API est inaccessible, on évite le crash de l'UI
            st.error(f"Erreur de connexion à l'API RAG : {e}")
            collections_disponibles = [st.session_state.rag_config["collection"]]

        # 2. Gestion de l'index par défaut du selectbox
        collection_actuelle = st.session_state.rag_config["collection"]
        try:
            index_par_defaut = collections_disponibles.index(collection_actuelle)
        except ValueError:
            # Si la collection en session n'existe plus, on sélectionne la première de la liste
            index_par_defaut = 0

        # 3. Affichage du menu déroulant
        st.session_state.rag_config["collection"] = st.selectbox(
            "Collection ChromaDB", 
            options=collections_disponibles,
            index=index_par_defaut
        )
        st.session_state.rag_config["model"] = st.text_input(
            "Modèle LLM", 
            value=st.session_state.rag_config["model"]
        )
        st.session_state.rag_config["n_results"] = st.number_input(
            "Nombre de chunks à récupérer (n_results)", 
            min_value=1, 
            max_value=1000, 
            value=st.session_state.rag_config["n_results"]
        )
        st.session_state.rag_config["doc_date_filter"] = st.text_input(
            "Filtre de date (doc_date_filter)", 
            value=st.session_state.rag_config["doc_date_filter"],
            help="Laissez vide pour ne pas filtrer par date."
        )

    with col2:
        st.session_state.rag_config["seuil"] = st.slider(
            "Seuil de pertinence", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(st.session_state.rag_config["seuil"]),
            step=0.05
        )
        st.session_state.rag_config["alpha"] = st.slider(
            "Équilibre Vectoriel / BM25 (alpha)", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(st.session_state.rag_config["alpha"]),
            step=0.05
        )
        st.session_state.rag_config["use_hyde"] = st.toggle(
            "Utiliser HyDE (Hypothetical Document Embeddings)", 
            value=st.session_state.rag_config["use_hyde"]
        )
        st.session_state.rag_config["use_expansion"] = st.toggle(
            "Expansion de requête (Synonymes)", 
            value=st.session_state.rag_config["use_expansion"]
        )

    st.divider()

    # -- Section 2 : Prompt Système --
    st.header("Prompt Système")
    st.session_state.system_prompt = st.text_area(
        "Modifiez le comportement du modèle (Session locale)",
        value=st.session_state.system_prompt,
        height=350
    )
    
    if st.button("Réinitialiser les paramètres par défaut", type="primary"):
        st.session_state.pop("rag_config", None)
        st.session_state.pop("system_prompt", None)
        st.rerun()

if __name__ == "__main__":
    render_config_page()