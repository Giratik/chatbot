import chromadb
from chromadb.utils import embedding_functions
import os
import json
import re

URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=URL_OLLAMA + "/api/embeddings",
    model_name="nomic-embed-text"
)

def get_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

def get_collection():
    client = get_client()
    return client.get_collection(
        name="base_connaissances_globale_acronymes",
        embedding_function=ollama_ef
    )



def recherche_lexique(query: str, collection, n_results=5, seuil=0.6):
    # 1. Match exact sur l'acronyme
    exact = collection.get(where={"acronyme": query.upper()})
    if exact["documents"]:
        print("✅ Match exact acronyme")
        return {
            "documents": [exact["documents"]],
            "metadatas": [exact["metadatas"]],
            "distances": [[0.0] * len(exact["documents"])]
        }

    # 2. Match exact sur la signification
    exact_sig = collection.get(where={"signification": query})
    if exact_sig["documents"]:
        print("✅ Match exact signification")
        return {
            "documents": [exact_sig["documents"]],
            "metadatas": [exact_sig["metadatas"]],
            "distances": [[0.0] * len(exact_sig["documents"])]
        }

    # 3. Fallback vectoriel avec seuil
    print("🔍 Recherche sémantique...")
    results = collection.query(query_texts=[query], n_results=n_results)

    filtered = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist <= seuil:
            filtered["documents"][0].append(doc)
            filtered["metadatas"][0].append(meta)
            filtered["distances"][0].append(dist)

    #for doc, meta, dist in zip(filtered["documents"][0], filtered["metadatas"][0], filtered["distances"][0]):
    #    print(f"{meta['acronyme']} → {meta['signification']} (distance: {dist:.4f})")

    return filtered



def extraire_acronymes(texte: str) -> list[str]:
    # Capture les mots en majuscules de 2+ lettres (CODIR, GPEC, R&D...)
    return re.findall(r'\b[A-Z][A-Z0-9&]{1,}\b', texte)

def recherche_depuis_texte(texte: str, collection, n_results=3):
    acronymes = extraire_acronymes(texte)
    resultats = {}

    # 1. Résolution des acronymes détectés
    for acr in acronymes:
        exact = collection.get(where={"acronyme": acr})
        if exact["documents"]:
            resultats[acr] = exact["metadatas"][0]["signification"]

    # 2. Recherche sémantique sur la phrase entière en complément
    vecto = collection.query(query_texts=[texte], n_results=n_results)
    for meta, dist in zip(vecto["metadatas"][0], vecto["distances"][0]):
        acr = meta["acronyme"]
        if acr not in resultats and dist < 0.3:
            resultats[acr] = meta["signification"]

    return resultats


#def traiter_pdf(path: str, collection):
#    # 1. Extraire le texte
#    with pdfplumber.open(path) as pdf:
#        texte = "\n".join(page.extract_text() for page in pdf.pages)
#
#    # 2. Même fonction
#    return recherche_depuis_texte(texte, collection)