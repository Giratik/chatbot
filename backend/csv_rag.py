import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import Dict


# --- TA CONFIGURATION CHROMA NATIVE ---
# (J'ai mis host.docker.internal par défaut pour éviter l'erreur de connexion Docker)
URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Clients Chroma : un par session (en mémoire, pas de persistance disque)
# Cela évite l'accumulation de dossiers dans ./chroma_db
_csv_clients: Dict[str, chromadb.Client] = {}

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=URL_OLLAMA + "/api/embeddings",
    model_name="nomic-embed-text"
)


def get_csv_client(session_id: str = "default") -> chromadb.Client:
    """Obtient ou crée un client Chroma en mémoire pour cette session."""
    if session_id not in _csv_clients:
        _csv_clients[session_id] = chromadb.Client()
        print(f"Nouveau client Chroma créé pour la session {session_id}.")
    return _csv_clients[session_id]


def get_csv_collection_name(session_id: str = "default") -> str:
    return f"csv_knowledge_base_{session_id}"


def delete_csv_session(session_id: str = "default") -> None:
    """Supprime la collection et libère le client pour cette session."""
    client = get_csv_client(session_id)
    name = get_csv_collection_name(session_id)
    try:
        client.delete_collection(name=name)
        print(f"Collection '{name}' supprimée avec succès.")
    except Exception as e:
        print(f"Info: Collection '{name}' n'existait pas ou impossible à supprimer.")
    
    # Libération du client pour cette session (garbage collection)
    if session_id in _csv_clients:
        del _csv_clients[session_id]
        print(f"Client Chroma libéré pour la session {session_id}.")


def process_csv_file(file_path: str, session_id: str = "default"):
    client = get_csv_client(session_id)
    collection_name = get_csv_collection_name(session_id)

    # 1. NETTOYAGE : On supprime l'ancienne base si elle existe
    try:
        client.delete_collection(name=collection_name)
        print(f"Ancienne collection '{collection_name}' supprimée.")
    except Exception:
        # Ignorer silencieusement si la collection n'existe pas
        pass
        
    # 2. CRÉATION : On recrée une base totalement vierge
    collection = client.create_collection(
        name=collection_name, 
        embedding_function=ollama_ef
    )

    # 3. Lecture du NOUVEAU CSV
    loader = CSVLoader(file_path, encoding="utf-8")
    docs = loader.load_and_split()
    
    # 4. Préparation et ajout natif
    documents = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [str(uuid.uuid4()) for _ in docs] 
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    # vérifier la création de la collection et le nombre de documents ajoutés
    print(f"Collection '{collection_name}' créée avec {len(docs)} documents.")
    return len(docs)
