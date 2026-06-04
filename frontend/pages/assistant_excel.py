# frontend/pages/assistant_excel.py

import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Assistant Data & Graphiques", page_icon="📊", layout="wide")

# Importer et appeler la fonction encapsulée
from renders.excel_analyst_ui import render_excel_analyst

render_excel_analyst(title="📊 Assistant Data & Graphiques")
