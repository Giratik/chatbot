"""
ui/chunks.py
────────────
Panneau de visualisation des chunks récupérés lors de la dernière requête.
"""

import streamlit as st


def render_chunk_card(chunk: dict) -> None:
    """Affiche une card pour un chunk dans le panneau de visualisation."""
    chunk_type = chunk["type"]
    source = chunk["source"]
    doc = chunk["document"]
    hybrid = chunk["hybrid_score"]
    vecto = chunk["vecto_distance"]
    bm25 = chunk["bm25_score"]
    doc_date = chunk.get("doc_date", "")
    rerank = chunk.get("rerank_score", 0.0)

    if chunk_type == "pdf":
        tag_cls, card_cls, label = "tag-pdf", "pdf", "📄 PDF"
    elif chunk_type == "lexique":
        tag_cls, card_cls, label = "tag-semantic", "semantic", "📚 Lexique"
    else:
        tag_cls, card_cls, label = "tag-semantic", "semantic", "📋 Document"

    date_badge = f"<span class='score'>📅 {doc_date}</span> " if doc_date else ""

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <span class="tag {tag_cls}">{label}</span>
        {date_badge}<span class="score">H:{hybrid:.3f} | V:{vecto:.4f} | B:{bm25:.3f} | R:{rerank:.3f}</span><br>
        <strong>{source}</strong>
        <div style="color:var(--text2);font-size:0.78rem;margin-top:0.4rem;line-height:1.4;">{doc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_chunks_panel() -> None:
    """Affiche la colonne de visualisation des chunks récupérés."""
    st.markdown("### 📦 Chunks Récupérés")

    chunks = st.session_state.get("last_chunks", [])
    if not chunks:
        st.info("Posez une question pour voir les chunks récupérés ici.")
        return

    st.caption(f"{len(chunks)} chunk(s) récupérés")
    st.markdown("---")

    for i, chunk in enumerate(chunks):
        with st.expander(f"**Chunk {i+1}** — {chunk['source'][:40]}...", expanded=(i == 0)):
            render_chunk_card(chunk)
            st.markdown("**Scores détaillés :**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Hybride", f"{chunk['hybrid_score']:.3f}")
            c2.metric("Vectoriel", f"{chunk['vecto_distance']:.4f}")
            c3.metric("BM25", f"{chunk['bm25_score']:.3f}")
            c4 = st.columns(4)[3]
            c4.metric("Rerank", f"{chunk.get('rerank_score', 0.0):.3f}")