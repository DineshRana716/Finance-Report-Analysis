import os
import time

import fitz  # PyMuPDF
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- Settings ----------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]   # tried in order
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 6
CHROMA_PATH = "chroma_db"

# ---------- Shared objects (created once on import) ----------
embedder = SentenceTransformer(EMBED_MODEL)
chroma = chromadb.PersistentClient(CHROMA_PATH)
llm = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ---------- Ingestion ----------
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return [{"page": n, "text": p.get_text()} for n, p in enumerate(doc, start=1)]


def make_chunks(pages, company, year, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for p in pages:
        for i, piece in enumerate(splitter.split_text(p["text"])):
            if len(piece.strip()) < 30:
                continue
            chunks.append({
                "id": f"{company}-{year}-p{p['page']}-{i}",
                "text": piece,
                "metadata": {"company": company, "year": year, "page": p["page"]},
            })
    return chunks


def collection_name(chunk_size, chunk_overlap):
    # one collection per chunking setup, so experiments don't overwrite each other
    return f"reports_c{chunk_size}_o{chunk_overlap}"


def ingest(pdf_path, company, year, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    pages = extract_pages(pdf_path)
    chunks = make_chunks(pages, company, year, chunk_size, chunk_overlap)
    texts = [c["text"] for c in chunks]
    embs = embedder.encode(texts, batch_size=32, normalize_embeddings=True)

    col = chroma.get_or_create_collection(
        collection_name(chunk_size, chunk_overlap), metadata={"hnsw:space": "cosine"}
    )
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embs.tolist(),
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


# ---------- Retrieval ----------
def retrieve(question, k=TOP_K, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
             company=None, year=None):
    col = chroma.get_collection(collection_name(chunk_size, chunk_overlap))

    conds = []
    if company:
        conds.append({"company": company})
    if year:
        conds.append({"year": year})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})

    q_emb = embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[q_emb], n_results=k, where=where)
    return [
        {"text": d, "distance": dist, **m}
        for d, dist, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])
    ]


# ---------- Generation ----------
PROMPT = """You are a financial analyst assistant. Answer using ONLY the context below.
- Cite sources like (Microsoft FY2026, p.45).
- Quote numbers exactly, with units and currency.
- If the answer is not in the context, say "I couldn't find this in the uploaded reports."

Context:
{context}

Question: {question}
Answer:"""


def generate(prompt, temperature=0):
    last_error = None
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
                break   # other error (e.g. 404): try the next model
    raise last_error


def ask(question, k=TOP_K, **retrieve_kwargs):
    chunks = retrieve(question, k, **retrieve_kwargs)
    context = "\n\n".join(
        f"[{c['company']} FY{c['year']}, p.{c['page']}]\n{c['text']}" for c in chunks
    )
    return generate(PROMPT.format(context=context, question=question)), chunks