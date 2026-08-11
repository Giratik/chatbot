import streamlit as st
import requests
import json


# Assurez-vous d'importer votre fonction client
from plugins.APIclient import get_registry

def obtenir_description_collections_dynamique() -> str:
    """
    Récupère dynamiquement les noms et descriptions des collections
    depuis la collection '_registry' de Qdrant via l'API backend.
    """
    try:
        import streamlit as st

        # Get backend URL from session state or use default
        backend_url = st.session_state.get("backend_url", "http://10.75.12.5:8000")

        # Appel à l'API backend pour récupérer le registre
        #response = get_registry()
        #response = requests.get(f"{backend_url}/rag/registry")
        #response.raise_for_status()
        #data = response.json()
        #registry_entries = data.get("registry", [])
        response = get_registry()
        registry_entries = response.get("registry", [])

        # Filtrer les entrées valides et extraire nom + description
        collections_dispos = []
        for entry in registry_entries:
            if entry.get("collection_name") and entry.get("description"):
                collections_dispos.append({
                    "nom": entry["collection_name"],
                    "description": entry["description"]
                })

        # Si le registre est vide, utiliser des valeurs par défaut
        if not collections_dispos:
            return "Nom de la collection dans laquelle chercher. (Attention: aucune collection disponible dans le registre)."

        # Construction de la chaîne de texte détaillée pour le LLM
        texte_description = "Nom de la collection dans laquelle chercher. Voici les choix obligatoires :\n"
        for col in collections_dispos:
            texte_description += f"- '{col['nom']}' : à utiliser pour des recherches concernant {col['description']}.\n"

        return texte_description

    except Exception as e:
        print(f"Erreur lors de la récupération du registre: {str(e)}")
        # Fallback de sécurité générique en cas d'erreur de connexion à Qdrant
        return (
            "erreur. qdrant est injoinable"
        )

def build_qdrant_tools():
    # On génère la description dynamique
    description_dynamique = obtenir_description_collections_dynamique()
    
    # On insère cette description dans le schéma
    outils = [
        {
            "type": "function",
            "function": {
                "name": "rechercher_dans_qdrant",
                "description": "Recherche des informations dans la base de données de l'entreprise.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": description_dynamique # ⬅️ INJECTION ICI
                        },
                        "query": {
                            "type": "string",
                            "description": "La requête ou les mots-clés optimisés pour la recherche."
                        }
                    },
                    "required": ["collection_name", "query"]
                }
            }
        }
    ]
    return outils





st.set_page_config(page_title="Test Tool Calling", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Débogueur de Tool Calling (Connecté au vrai Backend)")
st.markdown("Cette page teste le comportement du LLM et interroge **votre véritable base Qdrant** via votre route API `/rag/search`.")

with st.sidebar:
    st.header("⚙️ Configuration")
    ollama_url = st.text_input("URL Ollama", value="http://10.75.12.5:11434", help="L'adresse de votre serveur Ollama.")
    backend_url = st.text_input("URL Backend API", value="http://10.75.12.5:8000", help="L'adresse de votre backend FastAPI.")
    modele = st.text_input("Modèle", value="gemma4:e4b")

# Définition de l'outil par défaut
default_tools = build_qdrant_tools()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Le schéma de l'outil (Tools)")
    tools_str = st.text_area("Modifiez la description pour tester comment le LLM réagit :", 
                             value=json.dumps(default_tools, indent=4, ensure_ascii=False), 
                             height=350)
    
    st.subheader("2. La question de l'utilisateur")
    user_prompt = st.text_input("Posez une question :", value="Quel est le numéro de téléphone de Jean Dupont ?")
    
    tester = st.button("🚀 Tester le pipeline complet", use_container_width=True)

with col2:
    st.subheader("3. Résultat et Exécution")
    if tester:
        if not user_prompt:
            st.warning("Veuillez entrer une question.")
        else:
            try:
                tools_json = json.loads(tools_str)
            except json.JSONDecodeError:
                st.error("Erreur de syntaxe JSON dans le schéma de l'outil.")
                st.stop()

            payload_ollama = {
                "model": modele,
                "messages": [{"role": "user", "content": user_prompt}],
                "tools": tools_json,
                "stream": False
            }

            st.markdown("### 🔄 Étape 1 : Réflexion du LLM")
            with st.spinner("Le LLM analyse la question..."):
                try:
                    # Appel Ollama pour vérifier s'il veut un outil
                    response = requests.post(f"{ollama_url}/api/chat", json=payload_ollama)
                    response.raise_for_status()
                    resultat = response.json()
                    
                    message_assistant = resultat.get("message", {})
                    tool_calls = message_assistant.get("tool_calls", [])
                    
                    if tool_calls:
                        st.success("✅ Le LLM a décidé d'utiliser un outil !")
                        for tc in tool_calls:
                            nom_fonction = tc.get("function", {}).get("name")
                            arguments = tc.get("function", {}).get("arguments", {})
                            
                            st.markdown(f"**🛠️ Outil appelé :** `{nom_fonction}`")
                            st.markdown("**Arguments générés :**")
                            st.json(arguments)
                            
                            if nom_fonction == "rechercher_dans_qdrant":
                                col = arguments.get("collection_name", "INCONNUE")
                                q = arguments.get("query", "INCONNUE")
                                
                                st.markdown("---")
                                st.markdown("### 🔄 Étape 2 : Interrogation du vrai backend (Qdrant)")
                                
                                vrai_contexte = ""
                                
                                # Véritable appel à l'API backend /rag/search
                                with st.spinner(f"Recherche de '{q}' dans '{col}'..."):
                                    try:
                                        payload_search = {
                                            "collection_name": col,
                                            "query": q,
                                            "model": modele,
                                            "n_results": 3,
                                            "seuil": 0.5,
                                            "alpha": 0.5
                                        }
                                        search_resp = requests.post(f"{backend_url}/rag/search", json=payload_search)
                                        search_resp.raise_for_status()
                                        search_data = search_resp.json()
                                        
                                        contexts = search_data.get("contexts", [])
                                        
                                        if contexts:
                                            vrai_contexte = "\n\n".join(contexts)
                                            st.success(f"✅ {len(contexts)} extraits récupérés depuis la base Qdrant !")
                                            with st.expander("Voir les extraits trouvés (Contexte)"):
                                                st.write(vrai_contexte)
                                        else:
                                            vrai_contexte = "Aucune information pertinente n'a été trouvée dans la base de données."
                                            st.warning("⚠️ Qdrant n'a trouvé aucun résultat pertinent.")
                                            
                                    except requests.exceptions.ConnectionError:
                                        st.error(f"Impossible de se connecter au backend sur {backend_url}. Le serveur FastAPI est-il lancé ?")
                                        st.stop()
                                    except Exception as e:
                                        st.error(f"Erreur lors de la recherche backend : {e}")
                                        st.stop()
                                
                                st.markdown("---")
                                st.markdown("### 🔄 Étape 3 : Rédaction de la réponse finale")
                                
                                # Préparation du 2ème appel : on donne au LLM l'historique complet avec les vraies données
                                messages_historique = [
                                    {"role": "user", "content": user_prompt},
                                    message_assistant,  # Le tool_call généré par le LLM
                                    {"role": "tool", "content": vrai_contexte} # La vraie réponse de Qdrant
                                ]
                                
                                payload_final = {
                                    "model": modele,
                                    "messages": messages_historique,
                                    "stream": False
                                }
                                
                                with st.spinner("Le LLM rédige la réponse finale en s'appuyant sur Qdrant..."):
                                    resp2 = requests.post(f"{ollama_url}/api/chat", json=payload_final)
                                    resp2.raise_for_status()
                                    reponse_finale = resp2.json().get("message", {}).get("content", "")
                                    
                                    st.success("**💬 Réponse finale générée pour l'utilisateur :**")
                                    st.write(reponse_finale)

                    else:
                        st.warning("❌ Le LLM n'a appelé aucun outil. Il a répondu directement avec ses connaissances.")
                        st.success("**💬 Réponse du modèle :**")
                        st.write(message_assistant.get("content", ""))
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Impossible de se connecter à Ollama sur {ollama_url}. Vérifiez que le serveur est lancé.")
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à Ollama : {e}")