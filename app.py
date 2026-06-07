"""
app.py
Streamlit chat interface for the IBU Regulatory Chatbot.

Run locally:    streamlit run app.py
Deploy:         push to GitHub, connect to Streamlit Cloud
"""

import streamlit as st
from rag_pipeline import ask

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBU Regulatory Chatbot",
    page_icon="🎓",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main { max-width: 800px; }

    /* Chat message styling */
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    /* Source card */
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #1f77b4;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
    }
    .source-card-xref {
        background: #fff8e1;
        border-left: 3px solid #f59e0b;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
    }
    /* Pin every child to the same base size so Streamlit's p/div styles can't override */
    .source-card *, .source-card-xref * {
        font-size: 13px;
        line-height: 1.5;
        font-family: inherit;
    }
    .source-title {
        font-weight: 600;
        color: #1f77b4;
        font-size: 13px;
    }
    .source-meta {
        color: #666;
        font-size: 12px;
        margin-top: 2px;
    }
    .source-text {
        color: #333;
        font-size: 13px;
        margin-top: 6px;
    }

    /* Language badge */
    .lang-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        font-weight: 600;
        margin-left: 8px;
    }
    .lang-en { background: #dbeafe; color: #1e40af; }
    .lang-bs { background: #dcfce7; color: #166534; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
if "messages" not in st.session_state:
    # messages: list of {role, content, sources, language}
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    # conversation_history: list of {role, content} — passed to the LLM
    st.session_state.conversation_history = []


# ── Helper ─────────────────────────────────────────────────────────────────────
_NOT_FOUND_EN = "This information was not found in the available IBU regulatory documents."
_NOT_FOUND_BS = "Ova informacija nije pronađena u dostupnim propisima IBU-a."

def is_not_found(answer: str) -> bool:
    """Returns True when the LLM gave a 'not found' response."""
    a = answer.strip()
    return a == _NOT_FOUND_EN or a == _NOT_FOUND_BS


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎓 IBU Regulatory Chatbot")
st.caption(
    "Ask questions about IBU regulations in English or Bosnian. "
    "Answers are grounded in official IBU regulatory documents."
)
st.divider()


# ── Render existing chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"] == "assistant":
            # Language badge next to answer
            lang      = msg.get("language", "en")
            lang_text = "🇧🇦 Bosnian" if lang == "bs" else "🇬🇧 English"
            lang_cls  = "lang-bs" if lang == "bs" else "lang-en"
            st.markdown(
                f'{msg["content"]} '
                f'<span class="lang-badge {lang_cls}">{lang_text}</span>',
                unsafe_allow_html=True
            )

            # Expandable sources panel — only shown when a real answer was given
            sources = msg.get("sources", [])
            if sources and not is_not_found(msg["content"]):
                primary = [s for s in sources if not s.get("via_xref")]
                xrefs   = [s for s in sources if s.get("via_xref")]

                with st.expander(
                    f"📄 Sources ({len(primary)} articles"
                    + (f" + {len(xrefs)} cross-referenced" if xrefs else "")
                    + ")"
                ):
                    # Primary sources
                    for s in primary:
                        score_text = (
                            f"Score: {s['score']:.3f}"
                            if s.get("score") else ""
                        )
                        st.markdown(f"""
<div class="source-card">
    <div class="source-title">📌 {s['breadcrumb']}</div>
    <div class="source-meta">
        📁 {s['filename']}
        {"&nbsp;|&nbsp; 📄 Page " + str(s['page']) if s.get('page') else ""}
        {"&nbsp;|&nbsp; " + score_text if score_text else ""}
    </div>
    <div class="source-text">{s['raw_text']}...</div>
</div>
""", unsafe_allow_html=True)

                    # Cross-referenced sources
                    if xrefs:
                        st.markdown(
                            "**🔗 Cross-referenced articles "
                            "(automatically fetched):**"
                        )
                        for s in xrefs:
                            st.markdown(f"""
<div class="source-card-xref">
    <div class="source-title">🔗 {s['breadcrumb']}</div>
    <div class="source-meta">
        📁 {s['filename']}
        {"&nbsp;|&nbsp; 📄 Page " + str(s['page']) if s.get('page') else ""}
    </div>
    <div class="source-text">{s['raw_text']}...</div>
</div>
""", unsafe_allow_html=True)

        else:
            st.markdown(msg["content"])


# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask about IBU regulations... / Pitajte o propisima IBU-a..."
):
    # Show user message immediately
    st.session_state.messages.append({
        "role":    "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer with spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching regulations..."):
            result = ask(
                query=prompt,
                conversation_history=st.session_state.conversation_history,
                n_results=5
            )

        # Language badge
        lang      = result["language"]
        lang_text = "🇧🇦 Bosnian" if lang == "bs" else "🇬🇧 English"
        lang_cls  = "lang-bs" if lang == "bs" else "lang-en"

        st.markdown(
            f'{result["answer"]} '
            f'<span class="lang-badge {lang_cls}">{lang_text}</span>',
            unsafe_allow_html=True
        )

        # Sources panel — only shown when a real answer was given
        sources = result["sources"]
        if sources and not is_not_found(result["answer"]):
            primary = [s for s in sources if not s.get("via_xref")]
            xrefs   = [s for s in sources if s.get("via_xref")]

            with st.expander(
                f"📄 Sources ({len(primary)} articles"
                + (f" + {len(xrefs)} cross-referenced" if xrefs else "")
                + ")"
            ):
                for s in primary:
                    score_text = (
                        f"Score: {s['score']:.3f}"
                        if s.get("score") else ""
                    )
                    st.markdown(f"""
<div class="source-card">
    <div class="source-title">📌 {s['breadcrumb']}</div>
    <div class="source-meta">
        📁 {s['filename']}
        {"&nbsp;|&nbsp; 📄 Page " + str(s['page']) if s.get('page') else ""}
        {"&nbsp;|&nbsp; " + score_text if score_text else ""}
    </div>
    <div class="source-text">{s['raw_text']}...</div>
</div>
""", unsafe_allow_html=True)

                if xrefs:
                    st.markdown(
                        "**🔗 Cross-referenced articles "
                        "(automatically fetched):**"
                    )
                    for s in xrefs:
                        st.markdown(f"""
<div class="source-card-xref">
    <div class="source-title">🔗 {s['breadcrumb']}</div>
    <div class="source-meta">
        📁 {s['filename']}
        {"&nbsp;|&nbsp; 📄 Page " + str(s['page']) if s.get('page') else ""}
    </div>
    <div class="source-text">{s['raw_text']}...</div>
</div>
""", unsafe_allow_html=True)

    # Save to session state
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  result["answer"],
        "sources":  result["sources"],
        "language": result["language"]
    })


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
This chatbot answers questions about **IBU official regulations** using 
Retrieval-Augmented Generation (RAG).

**Knowledge base:**
- 12 IBU regulatory documents
- 560 article-level chunks
- Bosnian and English support

**How it works:**
1. Your question is encoded as a vector
2. The most relevant articles are retrieved from MongoDB Atlas
3. An LLM generates a grounded answer

**Supported languages:**
🇬🇧 English &nbsp;|&nbsp; 🇧🇦 Bosnian
    """)

    st.divider()

    # Clear conversation button
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages            = []
        st.session_state.conversation_history = []
        st.rerun()

    st.divider()
    st.caption(
        "Answers are based solely on official IBU regulatory documents. "
        "For official matters, always consult the university administration."
    )