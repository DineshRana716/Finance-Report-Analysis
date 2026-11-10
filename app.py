import html
from pathlib import Path

import streamlit as st

import rag

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED = ["pdf", "csv", "xlsx", "xls"]

st.set_page_config(
    page_title="Finance Report Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(1200px 500px at 10% -10%, #1c2b40 0%, #0e1621 45%); }
      .block-container { padding-top: 2.4rem; max-width: 1200px; }
      header[data-testid="stHeader"] { background: transparent; }
      .hero {
        padding: 0.2rem 0 1rem 0;
      }
      .hero h1 { font-size: 2rem; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
      .hero p { color: #9eb0c5; margin-top: 0; }
      .metric-card, .source-card, .empty-card {
        background: #172333;
        border: 1px solid #2a3c55;
        border-radius: 14px;
        padding: 0.9rem 1rem;
      }
      .source-card { margin-bottom: 0.6rem; }
      .source-card small { color: #9eb0c5; }
      .pill {
        display: inline-block;
        background: #24364d;
        color: #e7eef6;
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        margin-right: 0.35rem;
      }
      .found { color: #8fd4a4; }
      .missing { color: #f0b4b4; }
      div[data-testid="stChatMessage"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def boot():
    rag.get_embedder()
    return True


def save_upload(uploaded, company, year):
    folder = UPLOAD_DIR / rag.slug(company) / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    return dest


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)


def render_sources(chunks):
    if not chunks:
        return
    with st.expander("Sources used", expanded=False):
        for chunk in chunks:
            dist = chunk.get("distance")
            score = f"{1 - dist:.0%} match" if isinstance(dist, (int, float)) else ""
            snippet = html.escape((chunk.get("text") or "")[:900])
            st.markdown(
                f"""
                <div class="source-card">
                  <span class="pill">{html.escape(str(chunk.get('company', 'Unknown')))}</span>
                  <span class="pill">FY{html.escape(str(chunk.get('year', '?')))}</span>
                  <span class="pill">p.{html.escape(str(chunk.get('page', '?')))}</span>
                  <span class="pill">{html.escape(str(chunk.get('source', 'report')))}</span>
                  <span class="pill">{html.escape(score)}</span>
                  <div style="margin-top:0.55rem; white-space:pre-wrap; font-size:0.9rem;">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


init_state()
boot()

with st.sidebar:
    st.markdown("### Upload a report")
    st.caption("PDF annual reports, 10-Ks, or finance tables (CSV / Excel).")
    uploaded_files = st.file_uploader(
        "Finance files",
        type=ALLOWED,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    default_company, default_year = "", None
    if uploaded_files:
        default_company, default_year = rag.suggest_meta(uploaded_files[0].name)

    company = st.text_input("Company name", value=default_company, placeholder="e.g. Apple")
    year = st.number_input(
        "Fiscal year",
        min_value=1990,
        max_value=2100,
        value=int(default_year or 2026),
        step=1,
    )
    replace = st.checkbox("Replace if this file is already indexed", value=False)

    if st.button("Index files", type="primary", use_container_width=True, disabled=not uploaded_files):
        if not company.strip():
            st.error("Enter the company name before indexing.")
        else:
            progress = st.progress(0.0, text="Preparing…")
            errors = []
            total = len(uploaded_files)
            for i, uploaded in enumerate(uploaded_files, start=1):
                progress.progress((i - 1) / total, text=f"Indexing {uploaded.name}…")
                try:
                    path = save_upload(uploaded, company, int(year))
                    result = rag.ingest(
                        str(path),
                        company.strip(),
                        int(year),
                        force=replace,
                        source=uploaded.name,
                    )
                    if result["skipped"]:
                        st.info(f"{uploaded.name} is already indexed. Check replace to re-index.")
                    else:
                        st.success(f"Indexed {uploaded.name}: {result['chunks']} passages from {result['pages']} pages.")
                except Exception as exc:
                    errors.append(f"{uploaded.name}: {exc}")
            progress.progress(1.0, text="Done")
            for err in errors:
                st.error(err)
            st.rerun()

    st.divider()
    reports = rag.list_reports()
    companies = rag.list_companies()
    st.markdown("### Indexed library")
    if not reports:
        st.caption("No reports yet. Upload a filing to start.")
    else:
        for i, report in enumerate(reports):
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(
                    f"**{report['company']}** FY{report['year']}<br><small>{report['source']} · {report['chunks']} passages</small>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("✕", key=f"del-{i}", help="Remove this report"):
                    source = report["source"] if report["source"] != "indexed report" else None
                    rag.delete_report(report["company"], report["year"], source)
                    st.rerun()

    st.divider()
    st.markdown("### Ask about")
    company_filter = st.selectbox(
        "Company",
        options=["All companies"] + companies,
        index=0,
        disabled=not companies,
    )
    selected_company = None if company_filter == "All companies" else company_filter

    year_options = sorted(
        {
            r["year"]
            for r in reports
            if selected_company is None or r["company"] == selected_company
        }
    )
    year_filter = st.selectbox(
        "Fiscal year",
        options=["All years"] + year_options,
        index=0,
        disabled=not year_options,
    )
    selected_year = None if year_filter == "All years" else int(year_filter)

st.markdown(
    """
    <div class="hero">
      <h1>Finance Report Analyst</h1>
      <p>Upload any company's filings or finance tables, then ask questions. Answers come only from those documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary = rag.stats()
m1, m2, m3 = st.columns(3)
m1.metric("Companies", summary["companies"])
m2.metric("Reports", summary["reports"])
m3.metric("Indexed passages", summary["chunks"])

if summary["reports"] == 0:
    st.markdown(
        """
        <div class="empty-card">
          <strong>Get started</strong>
          <p style="color:#9eb0c5; margin-bottom:0;">
            Use the sidebar to upload a 10-K, annual report, or finance spreadsheet.
            After it is indexed, ask about revenue, margins, risks, or headcount.
            If a fact is not in the file, the assistant will say so instead of guessing.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    examples = [
        "What was total revenue?",
        "What was net income?",
        "What were the main risk factors?",
        "How many employees were there?",
    ]
    st.caption("Try a question")
    ex_cols = st.columns(len(examples))
    for col, question in zip(ex_cols, examples):
        if col.button(question, use_container_width=True):
            st.session_state.pending_question = question

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            badge = "Grounded in the uploaded reports" if message.get("found") else "Not found in the uploaded reports"
            klass = "found" if message.get("found") else "missing"
            st.markdown(f"<span class='{klass}'>{badge}</span>", unsafe_allow_html=True)
            if message.get("found"):
                render_sources(message.get("chunks") or [])

prompt = st.chat_input(
    "Ask a question about the indexed reports…",
    disabled=summary["reports"] == 0,
)
if st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt:
    if summary["reports"] == 0:
        st.warning("Index at least one report before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the uploaded reports…"):
                try:
                    result = rag.ask(
                        prompt,
                        company=selected_company,
                        year=selected_year,
                    )
                except RuntimeError as exc:
                    result = {"answer": str(exc), "chunks": [], "found": False}
                except Exception as exc:
                    result = {
                        "answer": f"Could not generate an answer: {exc}",
                        "chunks": [],
                        "found": False,
                    }

            st.markdown(result["answer"])
            badge = "Grounded in the uploaded reports" if result.get("found") else "Not found in the uploaded reports"
            klass = "found" if result.get("found") else "missing"
            st.markdown(f"<span class='{klass}'>{badge}</span>", unsafe_allow_html=True)
            if result.get("found"):
                render_sources(result.get("chunks") or [])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "chunks": result.get("chunks") or [],
                "found": result.get("found"),
            }
        )
