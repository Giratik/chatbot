# backend/main.py

"""
Module : Racine de l'application Backend (FastAPI)
Description : Point d'entrée principal de l'API. Initialise l'application FastAPI,
              configure les middlewares (CORS, etc.) et inclut les différents 
              routeurs (chat, files, data_analyst).
"""

from fastapi import FastAPI

# Importation des routeurs modulaires
from API_routes import rag, chat, excel_tool, files

app = FastAPI(title="API Chatbot")


# Inclusion des routeurs avec préfixes optionnels
app.include_router(chat.router)
app.include_router(excel_tool.router)
app.include_router(files.router)
app.include_router(rag.router)