import chromadb
from chromadb.utils import embedding_functions
import os
import json
import re
import pdfplumber  # ou pymupdf


URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=URL_OLLAMA + "/api/embeddings",
    model_name="nomic-embed-text"
)


def get_collection():
    client = chromadb.PersistentClient(path="./chromadb")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=URL_OLLAMA + "/api/embeddings",
        model_name="nomic-embed-text"
    )
    return client.get_collection(name="base_connaissances_globale_acronymes", embedding_function=ollama_ef)

def remplir_database_chroma():
    # 1. Connexion à la base de données
    client = chromadb.PersistentClient(path="./chromadb")
    
    # 3. Collection (Je vous conseille un nom générique vu qu'il y aura plusieurs sources)
    collection = client.get_or_create_collection(
        name="base_connaissances_globale_acronymes", 
        embedding_function=ollama_ef
    )

    # --- NOUVEAUTÉ : Dictionnaire des sources ---
    # Il vous suffit d'ajouter vos futurs fichiers JSON ici
    sources_a_ingérer = [
        {
            "chemin": "database/lexique.json", 
            "nom_source": "lexique_edp",
            "portee": "commun"
        }
    ]

    # 4. Boucle de traitement pour chaque source
    for source in sources_a_ingérer:
        chemin_json = source["chemin"]
        nom_source = source["nom_source"]
        portee = source["portee"]

        if not os.path.exists(chemin_json):
            print(f"❌ Erreur : Le fichier {chemin_json} est introuvable. On passe au suivant.")
            continue # Passe au fichier suivant sans tout bloquer

        print(f"\n⏳ Lecture et intégration de la source : {nom_source}...")
        with open(chemin_json, "r", encoding="utf-8") as f:
            donnees_json = json.load(f)

        # Cellule 5 — Ingestion
    collection.add(
        documents=[
        f"{v['acronyme']} {v['acronyme'].lower()} : {v['signification']}. "
        f"Également appelé {v['signification'].lower()}."
        for v in donnees_json
    ],
        metadatas=[
            {
                "acronyme": v["acronyme"],
                "signification": v["signification"]
            }
            for v in donnees_json
        ],
        ids=[str(i) for i in range(len(donnees_json))]  # ✅ ids générés
        # ✅ pas de embeddings= : Chroma les calcule via ollama_ef
    )

    print(f"\n📊 TERMINÉ ! La base ChromaDB contient maintenant {collection.count()} éléments au total.")



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