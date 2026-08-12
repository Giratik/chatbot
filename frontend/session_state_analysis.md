# Analyse des session_state dans le frontend

## Liste complète des session_state utilisés dans le frontend

### 1. **frontend/plugins/session_state.py** (Fichier principal de gestion)
- `session_id`: Identifiant unique de session
- `messages`: Liste des messages de conversation
- `processed_files`: Liste des fichiers déjà traités
- `think_mode`: Mode de raisonnement activé/désactivé
- `tables_info`: Informations sur les tables Excel
- `knowledge_ready`: Indique si la connaissance est prête pour l'analyse
- `last_file_id`: ID du dernier fichier traité
- `tables_data`: Données des tables Excel
- `excel_mode`: Mode Excel activé/désactivé
- `current_excel_file`: Nom du fichier Excel actuel
- `stage`: Étape actuelle du processus Excel
- `selected_sheet`: Feuille Excel sélectionnée
- `pending_excel_file`: Fichier Excel en attente (bytes)
- `pending_excel_name`: Nom du fichier Excel en attente
- `pending_sheet_names`: Liste des noms de feuilles en attente
- `pending_user_query`: Requête utilisateur en attente
- `query_to_execute`: Requête à exécuter
- `regenerate_request`: Demande de régénération
- `rag_config`: Configuration RAG (collection, modèle, etc.)
- `excel_bytes`: Bytes du fichier Excel
- `excel_name`: Nom du fichier Excel
- `excel_sheet`: Feuille Excel

### 2. **frontend/general_purpose_chat/general_purpose_chat_ui.py** (Interface principale)
- `knowledge_ready`: Vérification si prêt pour analyse
- `session_id`: Utilisé dans les requêtes API
- `think_mode`: Mode de raisonnement
- `tables_info`: Informations sur les tables
- `rag_config`: Configuration RAG
- `current_excel_file`: Fichier Excel actuel
- `tables_data`: Données des tables
- `messages`: Liste des messages
- `regenerate_request`: Gestion de la régénération
- `query_to_execute`: Exécution de requêtes différées
- `processed_files`: Fichiers traités
- `excel_mode`: Mode Excel
- `last_file_id`: ID du dernier fichier
- `stage`: Étape du processus
- `pending_sheet_names`: Feuilles en attente
- `selected_sheet`: Feuille sélectionnée
- `pending_excel_file`: Fichier Excel en attente
- `pending_excel_name`: Nom du fichier en attente
- `pending_user_query`: Requête en attente

### 3. **frontend/Main.py** (Point d'entrée principal)
- `is_dev`: Mode développeur
- `rag_config`: Configuration RAG

### 4. **frontend/plugins/excel_tools.py** (Outils Excel)
- `session_id`: Utilisé dans les requêtes API
- `pending_excel_file`: Fichier Excel en attente
- `pending_excel_name`: Nom du fichier en attente
- `selected_sheet`: Feuille sélectionnée
- `tables_info`: Informations sur les tables
- `knowledge_ready`: État de préparation
- `excel_sheet`: Feuille Excel
- `pending_user_query`: Requête utilisateur en attente
- `query_to_execute`: Requête à exécuter
- `stage`: Étape du processus
- `pending_sheet_names`: Feuilles en attente

### 5. **frontend/plugins/Sidebar.py** (Barre latérale)
- `messages`: Liste des messages
- `prompt_chunk_system`: Prompt système pour chunks
- `prompt_global_system`: Prompt système global
- `rag_config`: Configuration RAG

### 6. **frontend/debug_files/Rag_parameters_render.py** (Debug RAG)
- `prompt_chunk_system`: Prompt système pour chunks
- `prompt_global_system`: Prompt système global
- `rag_config`: Configuration RAG
- `collections_disponibles`: Collections disponibles

### 7. **frontend/pages/Configuration.py** (Configuration)
- `system_prompt`: Prompt système
- `rag_config`: Configuration RAG complète
- `collections_disponibles`: Collections disponibles

### 8. **frontend/debug_files/Chunks.py** (Debug Chunks)
- `messages`: Liste des messages
- `last_chunks`: Derniers chunks

### 9. **frontend/debug_files/Chatbot_RH_debug.py** (Debug RH)
- `messages`: Liste des messages

## Analyse de la structure actuelle

### Points clés:
1. **Centralisation partielle**: Le fichier `session_state.py` contient déjà une initialisation centralisée
2. **Utilisation dispersée**: Les session_state sont utilisés dans de nombreux fichiers différents
3. **Redondance**: Plusieurs fichiers accèdent aux mêmes session_state
4. **Complexité**: La gestion des états Excel est particulièrement complexe

### Recommandations pour la centralisation:
1. **Créer un module centralisé** qui gère tous les accès aux session_state
2. **Définir des constantes** pour les noms des session_state
3. **Créer des fonctions d'accès** pour encapsuler la logique
4. **Documenter** chaque session_state et son usage

## Prochaines étapes:
- [ ] Créer un module centralisé de gestion des session_state
- [ ] Mettre à jour tous les fichiers pour utiliser ce module
- [ ] Documenter la nouvelle architecture