# backend/main.py

import threading
from fastapi import FastAPI
from rag_engine import remplir_database_chroma

# Importation des routeurs modulaires
from routers import chat, data_analyst, files

app = FastAPI(title="API EDP Chatbot de Eau de Paris")

# Routine de démarrage asynchrone (ChromaDB)
def routine_demarrage():
    remplir_database_chroma()

@app.on_event("startup")
def startup_event():
    threading.Thread(target=routine_demarrage).start()

# Inclusion des routeurs avec préfixes optionnels
app.include_router(chat.router)
app.include_router(data_analyst.router)
app.include_router(files.router)