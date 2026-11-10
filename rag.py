import hashlib
import os
import re
import time
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 6
CHROMA_PATH = "chroma_db"
MAX_DISTANCE = 0.55
NOT_FOUND = "I couldn't find this in the uploaded reports."
MIN_TEXT_LEN = 30

_embedder = None
_chroma = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_chroma():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(CHROMA_PATH)
    return _chroma


def get_llm():
    global _llm
    if _llm is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        _llm = genai.Client(api_key=api_key)
    return _llm


def collection_name(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    return f"reports_c{chunk_size}_o{chunk_overlap}"


def get_collection(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    return get_chroma().get_or_create_collection(
        collection_name(chunk_size, chunk_overlap),
        metadata={"hnsw:space": "cosine"},
    )


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip()).strip("-").lower()
    return cleaned or "company"


def suggest_meta(filename: str):
    stem = Path(filename).stem
    year = None
    match = re.search(r"(19|20)\d{2}", stem)
    if match:
        year = int(match.group())
        stem = stem.replace(match.group(), " ")
    company = re.sub(r"[_\-]+", " ", stem).strip().title() or "Unknown"
    return company, year


def extract_pages(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".csv":
        return _extract_tabular(path, csv=True)
    if suffix in {".xlsx", ".xls"}:
        return _extract_tabular(path, csv=False)
    raise ValueError(f"Unsupported file type: {suffix}. Upload a PDF, CSV, or Excel file.")


def _extract_pdf(path: str):
    doc = fitz.open(path)
    pages = [{"page": n, "text": page.get_text()} for n, page in enumerate(doc, start=1)]
    doc.close()
    return pages


def _extract_tabular(path: str, csv: bool):
    import pandas as pd

    if csv:
        sheets = {"data": pd.read_csv(path)}
    else:
        sheets = pd.read_excel(path, sheet_name=None)

    pages = []
    page_no = 1
    for sheet_name, df in sheets.items():
        df = df.fillna("")
        if df.empty:
            continue
        step = 20
        for start in range(0, len(df), step):
            part = df.iloc[start : start + step]
            header = " | ".join(str(c) for c in part.columns)
            rows = [" | ".join(str(v) for v in row.tolist()) for _, row in part.iterrows()]
            text = (
                f"Sheet: {sheet_name}\nRows {start + 1}-{start + len(part)}\n"
                f"{header}\n" + "\n".join(rows)
            )
            pages.append({"page": page_no, "text": text})
            page_no += 1
    return pages


def make_chunks(pages, company, year, source, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source_hash = hashlib.md5(source.encode("utf-8")).hexdigest()[:8]
    prefix = f"{slug(company)}-{year}-{source_hash}"
    chunks = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page["text"])):
            if len(piece.strip()) < MIN_TEXT_LEN:
                continue
            chunks.append(
                {
                    "id": f"{prefix}-p{page['page']}-{i}",
                    "text": piece,
                    "metadata": {
                        "company": company,
                        "year": int(year),
                        "page": int(page["page"]),
                        "source": source,
                    },
                }
            )
    return chunks


def _report_where(company, year=None, source=None):
    conds = [{"company": company}]
    if year is not None:
        conds.append({"year": int(year)})
    if source:
        conds.append({"source": source})
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


def ingest(pdf_path, company, year, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, force=False, source=None):
    """Index a finance file. `pdf_path` may be PDF, CSV, or Excel."""
    source = source or Path(pdf_path).name
    company = (company or "").strip()
    if not company:
        raise ValueError("Company name is required.")
    year = int(year)

    col = get_collection(chunk_size, chunk_overlap)
    existing = col.get(where=_report_where(company, year, source), limit=1)
    if existing["ids"] and not force:
        return {"chunks": 0, "skipped": True, "source": source}

    if existing["ids"] and force:
        col.delete(where=_report_where(company, year, source))

    pages = extract_pages(pdf_path)
    chunks = make_chunks(pages, company, year, source, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(
            "No readable text was found. If this is a scanned PDF, use a text-based report."
        )

    texts = [c["text"] for c in chunks]
    embs = get_embedder().encode(texts, batch_size=32, normalize_embeddings=True)
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embs.tolist(),
        metadatas=[c["metadata"] for c in chunks],
    )
    return {"chunks": len(chunks), "skipped": False, "source": source, "pages": len(pages)}


def list_reports(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    col = get_collection(chunk_size, chunk_overlap)
    if col.count() == 0:
        return []
    data = col.get(include=["metadatas"])
    grouped = {}
    for meta in data["metadatas"] or []:
        company = meta.get("company") or "Unknown"
        year = meta.get("year")
        source = meta.get("source") or "indexed report"
        key = (company, year, source)
        grouped[key] = grouped.get(key, 0) + 1
    reports = [
        {"company": company, "year": year, "source": source, "chunks": count}
        for (company, year, source), count in grouped.items()
    ]
    reports.sort(key=lambda r: (str(r["company"]).lower(), r["year"] or 0, r["source"]))
    return reports


def list_companies(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    return sorted({r["company"] for r in list_reports(chunk_size, chunk_overlap)})


def delete_report(company, year=None, source=None, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    col = get_collection(chunk_size, chunk_overlap)
    col.delete(where=_report_where(company, year, source))


def stats(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    reports = list_reports(chunk_size, chunk_overlap)
    companies = {r["company"] for r in reports}
    return {
        "reports": len(reports),
        "companies": len(companies),
        "chunks": sum(r["chunks"] for r in reports),
    }


def retrieve(question, k=TOP_K, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, company=None, year=None):
    col = get_collection(chunk_size, chunk_overlap)
    total = col.count()
    if total == 0:
        return []

    conds = []
    if company:
        conds.append({"company": company})
    if year is not None:
        conds.append({"year": int(year)})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})

    q_emb = get_embedder().encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    n_results = min(k, total)
    res = col.query(query_embeddings=[q_emb], n_results=n_results, where=where)
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    if not docs:
        return []
    return [{"text": d, "distance": dist, **m} for d, dist, m in zip(docs, dists, metas)]


PROMPT = """You are a financial analyst assistant. Answer using ONLY the context below.
Rules:
- Use only facts that appear in the context. Do not use outside knowledge.
- Cite sources like (Company FY2026, p.45).
- Quote numbers exactly, with units and currency.
- If the context does not contain the answer, reply with exactly: {not_found}
- If the question is about a different company or period than the context, reply with exactly: {not_found}

Context:
{{context}}

Question: {{question}}
Answer:""".format(not_found=NOT_FOUND)


def generate(prompt, temperature=0):
    last_error = None
    llm = get_llm()
    for model in MODELS:
        for attempt in range(3):
            try:
                r = llm.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                return r.text
            except (errors.ServerError, errors.ClientError) as e:
                last_error = e
                if getattr(e, "code", None) in (429, 503):
                    time.sleep(5 * (attempt + 1))
                    continue
                break
    raise last_error


def ask(question, k=TOP_K, **retrieve_kwargs):
    question = (question or "").strip()
    if not question:
        return {"answer": "Please ask a question about an uploaded report.", "chunks": [], "found": False}

    chunks = retrieve(question, k, **retrieve_kwargs)
    relevant = [c for c in chunks if c.get("distance", 1) <= MAX_DISTANCE]
    if not relevant:
        return {"answer": NOT_FOUND, "chunks": chunks, "found": False}

    context = "\n\n".join(
        f"[{c.get('company', 'Unknown')} FY{c.get('year', '?')}, p.{c.get('page', '?')}, {c.get('source', 'report')}]\n{c['text']}"
        for c in relevant
    )
    prompt = PROMPT.replace("{context}", context).replace("{question}", question)
    answer = generate(prompt)
    found = NOT_FOUND.lower() not in (answer or "").strip().lower()
    return {"answer": answer, "chunks": relevant, "found": found}
