"""
app.py -- LegalLens AI v3
Premium dark legal-tech UI redesign.
"""

import streamlit as st
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not needed on HuggingFace Spaces (uses os.environ directly)

from src.ingestion import extract_text_from_pdf, extract_text_from_image
from src.processing import chunk_text
from src.analysis import load_summarizer, load_risk_detector, analyze_document
from src.rag import ContractRAG
from src.report import generate_pdf_report, REPORTLAB_AVAILABLE

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LegalLens AI",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS -- Premium Dark Legal-Tech Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D0F14;
    color: #E8E6E0;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stMarkdownContainer"] a.anchor-link { display: none !important; }
.block-container { padding-top: 2rem !important; max-width: 1200px; }

[data-testid="stSidebar"] {
    background: #0A0C10 !important;
    border-right: 1px solid #1E2130;
}
[data-testid="stSidebar"] * { color: #A0A8BC !important; }
[data-testid="stSidebar"] .sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #C9A84C !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .step-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1A1D28;
}
[data-testid="stSidebar"] .step-num {
    background: #1A2035;
    color: #C9A84C !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    min-width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
}
[data-testid="stSidebar"] .step-label {
    color: #E8E6E0 !important;
    font-weight: 500;
    font-size: 0.82rem;
}

.ll-hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1E2130;
    margin-bottom: 2rem;
}
.ll-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #E8E6E0;
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
}
.ll-wordmark span { color: #C9A84C; }
.ll-tagline {
    font-size: 0.9rem;
    color: #3A4050;
    font-weight: 300;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 8px;
}
.ll-version-pill {
    display: inline-block;
    background: #1A2035;
    color: #C9A84C;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #2A3050;
    vertical-align: middle;
    margin-left: 12px;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #C9A84C;
    margin-bottom: 10px;
    display: block;
}

.file-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #111318;
    border: 1px solid #2A3050;
    border-radius: 8px;
    padding: 8px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #A0A8BC;
    margin-bottom: 1rem;
}
.file-badge .dot {
    width: 7px; height: 7px;
    background: #C9A84C;
    border-radius: 50%;
}

.ll-card {
    background: #111318;
    border: 1px solid #1E2130;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    color: #A0A8BC !important;
    line-height: 1.75;
    font-size: 0.9rem;
}

.risk-clean {
    background: #0B1A12;
    border: 1px solid #1A3A24;
    border-left: 3px solid #2E7D32;
    border-radius: 8px;
    padding: 14px 18px;
    color: #4CAF50 !important;
    font-size: 0.88rem;
    font-weight: 500;
}
.risk-alert {
    background: #1A0F0F;
    border: 1px solid #3A1A1A;
    border-left: 3px solid #C62828;
    border-radius: 8px;
    padding: 14px 18px;
    color: #EF5350 !important;
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 16px;
}

.risk-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 4px;
    text-transform: uppercase;
}
.risk-badge-high   { background: #2A0F0F; color: #FF5252; border: 1px solid #4A1A1A; }
.risk-badge-medium { background: #1A1500; color: #FFB300; border: 1px solid #3A2F00; }
.risk-badge-low    { background: #0A1525; color: #42A5F5; border: 1px solid #1A2A45; }

.conf-bar-wrap {
    background: #1A1D28;
    border-radius: 3px;
    height: 4px;
    margin-top: 8px;
    margin-bottom: 12px;
    overflow: hidden;
}
.conf-bar-fill-high   { height: 4px; border-radius: 3px; background: #C62828; }
.conf-bar-fill-medium { height: 4px; border-radius: 3px; background: #F57F17; }
.conf-bar-fill-low    { height: 4px; border-radius: 3px; background: #1565C0; }

.snippet-quote {
    background: #0D0F14;
    border-left: 2px solid #2A3050;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #5A6070 !important;
    line-height: 1.6;
    margin: 10px 0;
}

.stat-box {
    background: #111318;
    border: 1px solid #1E2130;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #C9A84C;
    line-height: 1;
}
.stat-label {
    font-size: 0.68rem;
    color: #3A4050;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 6px;
    font-family: 'DM Mono', monospace;
}

.chat-msg-user {
    background: #1A2035;
    border: 1px solid #2A3050;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 15%;
    color: #C9D0E0 !important;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chat-msg-bot {
    background: #111318;
    border: 1px solid #1E2130;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 15% 8px 0;
    color: #A0A8BC !important;
    font-size: 0.88rem;
    line-height: 1.6;
}
.chat-sender {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.chat-sender-user { color: #C9A84C !important; }
.chat-sender-bot  { color: #3A4050 !important; }

.citation-card {
    background: #0A0C10;
    border: 1px solid #1E2130;
    border-left: 2px solid #C9A84C;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.78rem;
    color: #3A4050 !important;
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
}

.export-feature {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #1E2130;
    color: #5A6070 !important;
    font-size: 0.85rem;
}
.export-feature-icon { color: #C9A84C !important; min-width: 16px; }

.landing-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 6rem 2rem;
    text-align: center;
}
.landing-glyph { font-size: 4rem; margin-bottom: 1.5rem; opacity: 0.15; }
.landing-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #2A3040;
    margin-bottom: 8px;
}
.landing-sub {
    font-size: 0.78rem;
    color: #1E2530;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
}

.stButton > button {
    background: #C9A84C !important;
    color: #0D0F14 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #E0BC60 !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.2) !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background: #0D0F14;
    border-bottom: 1px solid #1E2130;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #3A4050 !important;
    padding: 10px 20px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #C9A84C !important;
    border-bottom: 2px solid #C9A84C !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #111318 !important;
    border: 1px solid #2A3050 !important;
    border-radius: 8px !important;
    color: #E8E6E0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus { border-color: #C9A84C !important; }

[data-testid="stFileUploader"] {
    background: #111318 !important;
    border: 1.5px dashed #2A3050 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] {
    background: #111318 !important;
    border: 1px solid #1E2130 !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #A0A8BC !important;
    font-size: 0.85rem !important;
}

[data-testid="stProgress"] > div > div { background: #C9A84C !important; }
[data-testid="stStatus"] {
    background: #111318 !important;
    border: 1px solid #1E2130 !important;
    border-radius: 8px !important;
}
[data-testid="stInfo"] {
    background: #0A1525 !important;
    border: 1px solid #1A2A45 !important;
    color: #42A5F5 !important;
    border-radius: 8px !important;
}

hr { border-color: #1E2130 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D0F14; }
::-webkit-scrollbar-thumb { background: #2A3050; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #C9A84C; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing AI models...")
def get_models():
    summarizer    = load_summarizer()
    risk_detector = load_risk_detector()
    return summarizer, risk_detector


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "raw_text":      None,
    "chunks":        None,
    "summary":       None,
    "risks":         None,
    "rag":           None,
    "chat_history":  [],
    "doc_name":      "",
    "analysis_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">LegalLens</div>', unsafe_allow_html=True)
    st.caption("Contract Intelligence Platform")
    st.markdown("---")

    steps = [
        ("Ingest",   "OCR or native PDF extraction"),
        ("Chunk",    "Sentence-aware splitting"),
        ("Analyse",  "Legal-BERT + summarization"),
        ("Chat",     "RAG-powered Q&A"),
        ("Export",   "PDF risk report"),
    ]
    for i, (label, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-item">
            <div class="step-num">{i:02d}</div>
            <div>
                <div class="step-label">{label}</div>
                <div style="font-size:0.73rem;color:#2A3040">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq or Anthropic API Key",
        type="password",
        placeholder="gsk_... or sk-ant-...",
        value=default_key,
        help="Auto-loaded from Space secrets. Groq is free at console.groq.com",
    )
    st.markdown('<div style="font-size:0.72rem;color:#2A3040;margin-top:4px;font-family:DM Mono,monospace">Auto-loaded from secrets.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#2A3040;font-family:DM Mono,monospace">Built by Ardhi Gagan<br>LegalLens AI v3</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="ll-hero">
    <div class="ll-wordmark">Legal<span>Lens</span>
        <span class="ll-version-pill">v3.0</span>
    </div>
    <div class="ll-tagline">Contract Clarity &middot; Risk Intelligence &middot; AI Analysis</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
st.markdown('<span class="section-label">Upload Contract</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your contract here",
    type=["pdf", "png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file and (uploaded_file.name != st.session_state.doc_name):
    st.session_state.update({
        "raw_text": None, "chunks": None, "summary": None,
        "risks": None, "rag": None, "chat_history": [],
        "analysis_done": False, "doc_name": uploaded_file.name,
    })

if uploaded_file:
    st.markdown(f"""
    <div class="file-badge">
        <div class="dot"></div>
        {uploaded_file.name}&nbsp;&nbsp;
        <span style="color:#3A4050">{round(uploaded_file.size/1024,1)} KB</span>
    </div>
    """, unsafe_allow_html=True)

    # ── INGESTION ──────────────────────────────
    if st.session_state.raw_text is None:
        with st.status("Reading document...", expanded=True) as status:
            st.write("Extracting text...")
            file_bytes = uploaded_file.read()
            if uploaded_file.name.lower().endswith(".pdf"):
                raw_text = extract_text_from_pdf(file_bytes)
            else:
                raw_text = extract_text_from_image(file_bytes)
            st.session_state.raw_text = raw_text
            status.update(label="Document extracted", state="complete", expanded=False)

    with st.expander("View extracted text"):
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.78rem;color:#3A4050;line-height:1.7;white-space:pre-wrap">{st.session_state.raw_text[:3000]}{"..." if len(st.session_state.raw_text or "") > 3000 else ""}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ANALYSIS TRIGGER ──────────────────────
    if not st.session_state.analysis_done:
        col_btn, col_hint = st.columns([1, 3])
        with col_btn:
            run = st.button("Run Analysis")
        with col_hint:
            st.markdown('<div style="color:#2A3040;font-size:0.78rem;padding-top:12px;font-family:DM Mono,monospace">Summarization + Risk Detection + RAG Index</div>', unsafe_allow_html=True)

        if run:
            summarizer, risk_detector = get_models()
            progress = st.progress(0)
            with st.status("Running AI analysis...", expanded=True) as status:
                st.write("Chunking document...")
                chunks = chunk_text(st.session_state.raw_text)
                st.session_state.chunks = chunks
                progress.progress(20)

                st.write(f"Analyzing {len(chunks)} chunks...")
                summary, risks = analyze_document(chunks, summarizer, risk_detector)
                st.session_state.summary = summary
                st.session_state.risks   = risks
                progress.progress(75)

                st.write("Building semantic index...")
                rag = ContractRAG()
                rag.build_index(chunks)
                st.session_state.rag = rag
                progress.progress(100)

                status.update(label="Analysis complete", state="complete", expanded=False)

            st.session_state.analysis_done = True
            st.rerun()

    # ─────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────
    if st.session_state.analysis_done:
        risks = st.session_state.risks or []

        # ── STAT STRIP ────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        top_score = int(max((r["score"] for r in risks), default=0) * 100)
        risk_level = "HIGH" if top_score >= 80 else ("MEDIUM" if top_score >= 65 else ("LOW" if risks else "CLEAN"))
        rl_color   = "#FF5252" if risk_level == "HIGH" else ("#FFB300" if risk_level == "MEDIUM" else ("#42A5F5" if risk_level == "LOW" else "#4CAF50"))

        with s1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{len(st.session_state.chunks or [])}</div><div class="stat-label">Chunks Analysed</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{len(risks)}</div><div class="stat-label">Risks Detected</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{top_score}%</div><div class="stat-label">Top Confidence</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{rl_color};font-size:1.3rem">{risk_level}</div><div class="stat-label">Overall Risk</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TABS ──────────────────────────────
        tab_analysis, tab_chat, tab_export = st.tabs([
            "Risk Analysis", "Chat with Contract", "Export Report"
        ])

        # ── TAB 1: RISK ANALYSIS ──────────────
        with tab_analysis:
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown('<span class="section-label">Executive Summary</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="ll-card">{st.session_state.summary or "No summary generated."}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<span class="section-label">Risk Assessment</span>', unsafe_allow_html=True)

                if not risks:
                    st.markdown('<div class="risk-clean">No high-risk clauses detected in this document.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-alert">{len(risks)} potential risk(s) identified — review required</div>', unsafe_allow_html=True)

                    groups: dict[str, list] = {}
                    for r in risks:
                        groups.setdefault(r["type"], []).append(r)

                    for label, items in groups.items():
                        max_score = max(i["score"] for i in items)
                        pct       = int(max_score * 100)
                        sev       = "high" if max_score >= 0.80 else ("medium" if max_score >= 0.65 else "low")

                        with st.expander(f"{label}  —  {pct}%"):
                            st.markdown(f"""
                            <span class="risk-badge risk-badge-{sev}">{sev}</span>
                            <div class="conf-bar-wrap">
                                <div class="conf-bar-fill-{sev}" style="width:{pct}%"></div>
                            </div>
                            """, unsafe_allow_html=True)
                            for j, item in enumerate(items):
                                st.markdown(f'<div class="snippet-quote">{item["text_snippet"]}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#2A3040">Confidence: {int(item["score"]*100)}%</div>', unsafe_allow_html=True)
                                if j < len(items) - 1:
                                    st.markdown('<hr>', unsafe_allow_html=True)

        # ── TAB 2: CHAT ───────────────────────
        with tab_chat:
            st.markdown('<span class="section-label">Ask about your contract</span>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.78rem;color:#2A3040;margin-bottom:16px;font-family:DM Mono,monospace">Answers are grounded in the document with clause citations.</div>', unsafe_allow_html=True)

            if st.session_state.rag is None:
                st.info("Run analysis first to enable chat.")
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-msg-user"><div class="chat-sender chat-sender-user">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-msg-bot"><div class="chat-sender chat-sender-bot">LegalLens AI</div>{msg["answer"]}</div>', unsafe_allow_html=True)
                        if msg.get("citations"):
                            with st.expander(f"{len(msg['citations'])} source clause(s)"):
                                for k, c in enumerate(msg["citations"], 1):
                                    st.markdown(f'<div class="citation-card">Clause {k} &nbsp;|&nbsp; Relevance {int(c["score"]*100)}%<br><br>{c["text"][:300]}{"..." if len(c["text"])>300 else ""}</div>', unsafe_allow_html=True)

                question = st.chat_input("Ask anything about this contract...")
                if question:
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    with st.spinner("Searching contract..."):
                        answer, citations = st.session_state.rag.ask(
                            question, api_key=api_key if api_key else None
                        )
                    st.session_state.chat_history.append({
                        "role": "assistant", "answer": answer, "citations": citations
                    })
                    st.rerun()

                st.markdown('<span class="section-label" style="margin-top:20px;display:block">Suggested questions</span>', unsafe_allow_html=True)
                suggestions = [
                    "What are the termination conditions?",
                    "Are there automatic renewal clauses?",
                    "What are my obligations?",
                    "What happens if I breach this?",
                    "Who owns the intellectual property?",
                    "Are there any penalty clauses?",
                ]
                cols = st.columns(3)
                for i, s in enumerate(suggestions):
                    with cols[i % 3]:
                        if st.button(s, key=f"sug_{i}"):
                            st.session_state.chat_history.append({"role": "user", "content": s})
                            with st.spinner("Searching..."):
                                answer, citations = st.session_state.rag.ask(
                                    s, api_key=api_key if api_key else None
                                )
                            st.session_state.chat_history.append({
                                "role": "assistant", "answer": answer, "citations": citations
                            })
                            st.rerun()

                if st.session_state.chat_history:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Clear conversation"):
                        st.session_state.chat_history = []
                        st.rerun()

        # ── TAB 3: EXPORT ─────────────────────
        with tab_export:
            st.markdown('<span class="section-label">PDF Risk Report</span>', unsafe_allow_html=True)

            if not REPORTLAB_AVAILABLE:
                st.error("ReportLab not installed. Add `reportlab` to requirements.txt.")
            else:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    report_title = st.text_input(
                        "title",
                        value=st.session_state.doc_name or "Contract Analysis",
                        label_visibility="collapsed",
                        placeholder="Report title...",
                    )
                with col_b:
                    if st.button("Generate PDF"):
                        with st.spinner("Building report..."):
                            pdf_bytes = generate_pdf_report(
                                filename=report_title,
                                summary=st.session_state.summary or "",
                                risks=st.session_state.risks or [],
                                document_name=report_title,
                            )
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name=f"LegalLens_{report_title.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<span class="section-label">Report includes</span>', unsafe_allow_html=True)
                for f in [
                    "Document metadata and overall risk level",
                    "AI-generated executive summary",
                    "Risk overview table sorted by confidence",
                    "Detailed findings with evidence snippets",
                    "Legal disclaimer",
                ]:
                    st.markdown(f'<div class="export-feature"><span class="export-feature-icon">&#8212;</span><span>{f}</span></div>', unsafe_allow_html=True)

# ── LANDING ────────────────────────────────────
else:
    st.markdown("""
    <div class="landing-wrap">
        <div class="landing-glyph">&#9878;</div>
        <div class="landing-title">Upload a contract to begin</div>
        <div class="landing-sub">PDF &nbsp;&middot;&nbsp; PNG &nbsp;&middot;&nbsp; JPG &nbsp;&middot;&nbsp; Scanned documents supported</div>
    </div>
    """, unsafe_allow_html=True)
