"""
ui/styles.py
────────────
Injection du CSS global de l'application.
"""

import streamlit as st


def render_styles() -> None:
    """Injecte la feuille de style globale dans la page Streamlit."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500&display=swap');
:root {
    --surface: #e2e2e2;
    --surface2: #e2e2e282;
    --border: #2a2f40;
    --accent: #5b8dee;
    --accent2: #38e8b0;
    --danger: #e85b5b;
    --warn: #e8b05b;
    --text2: #7a84a0;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg);}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
h1{font-family:'Space Mono',monospace;color:var(--accent2)!important;letter-spacing:-1px;}
h2,h3{font-family:'Space Mono',monospace;color:var(--accent)!important;}
.stTextInput>div>div>input{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;}
.stSelectbox>div>div{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;}
.stButton>button{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:6px!important;font-family:'Space Mono',monospace!important;font-size:0.78rem!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
.result-card{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.75rem;}
.result-card.exact{border-left:3px solid var(--accent2);}
.result-card.semantic{border-left:3px solid var(--accent);}
.result-card.pdf{border-left:3px solid var(--warn, #e8b05b);}
.tag{display:inline-block;padding:1px 8px;border-radius:12px;font-size:0.7rem;font-family:'Space Mono',monospace;margin-right:4px;}
.tag-exact{background:#1a3a2a;color:var(--accent2);}
.tag-semantic{background:#1a2a3a;color:var(--accent);}
.tag-pdf{background:#3a2a1a;color:#e8b05b;}
.score{font-family:'Space Mono',monospace;font-size:0.75rem;color:var(--text2);}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-family:'Space Mono',monospace;}
.badge-ok{background:#1a3a2a;color:var(--accent2);border:1px solid #2a5a3a;}
.badge-err{background:#3a1a1a;color:var(--danger);border:1px solid #5a2a2a;}
.rewrite-badge{font-family:'Space Mono',monospace;font-size:0.7rem;color:var(--text2);margin-bottom:0.5rem;padding:2px 8px;border-left:2px solid var(--accent);background:var(--surface2);}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)