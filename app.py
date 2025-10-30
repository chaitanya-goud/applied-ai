# app.py
import os
import re
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

# =============== USER CONFIG ===============
EMBEDDINGS_FILE = "embeddings/reviewer_embeddings.npz"
METADATA_FILE   = "embeddings/reviewer_metadata.csv"
MODEL_NAME      = "all-MiniLM-L6-v2"  # must match the model used for embeddings
HARDCODED_GROQ_KEY = os.environ["GROQ_API_KEY"]  # <— put your key here
WEIGHTS = {"abstract": 0.40, "title": 0.30, "keywords": 0.20, "summary": 0.10}
# ==========================================

# ---- dependency setup (quiet) ----
def _safe_imports():
    try:
        import fitz  # PyMuPDF
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymupdf"], check=False)
        import fitz  # noqa: F401

    try:
        from groq import Groq
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "groq"], check=False)
        from groq import Groq  # noqa: F401

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=False)
        from sentence_transformers import SentenceTransformer  # noqa: F401

_safe_imports()
import fitz
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- helpers (UI-silent) ----------
def setup_groq_client() -> Groq | None:
    key = HARDCODED_GROQ_KEY.strip()
    if not key or not key.startswith("gsk_"):
        st.error("Invalid or missing Groq API key. Please set HARDCODED_GROQ_KEY in app.py.")
        return None
    try:
        return Groq(api_key=key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return ""


def extract_paper_metadata(client: Groq, text: str) -> dict | None:
    try:
        system_prompt = (
            "You are an expert scholarly metadata extractor. "
            "Return STRICT JSON ONLY with keys exactly: "
            '{"title": str, "authors": [str], "abstract": str, "keywords": [str], "summary": str}. '
            "Do not include any commentary or markdown fences."
        )

        user_prompt = f'''
Extract from the following academic paper text.
Return STRICT JSON with keys:
"title", "authors", "abstract", "keywords", "summary".

Paper text:
\"\"\"{text[:8000]}\"\"\"
Return ONLY JSON (no code fences).
'''.strip()

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            max_tokens=2048,
            temperature=0.1
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|```\s*$", "", raw, flags=re.IGNORECASE).strip()
        data = json.loads(raw)

        return {
            "title": (data.get("title") or "").strip(),
            "authors": data.get("authors") or [],
            "abstract": (data.get("abstract") or "").strip(),
            "keywords": data.get("keywords") or [],
            "summary": (data.get("summary") or "").strip(),
        }
    except Exception as e:
        st.warning(f"Metadata extraction failed: {e}")
        return None


def load_reviewer_embeddings(embeddings_file: str, metadata_file: str):
    if not (os.path.exists(embeddings_file) and os.path.exists(metadata_file)):
        st.warning("Embeddings or metadata file not found. Check the configured paths.")
        return None, None

    try:
        data = np.load(embeddings_file, allow_pickle=True)
        embeddings = {
            "title": data["title_embeddings"],
            "abstract": data["abstract_embeddings"],
            "keywords": data["keywords_embeddings"],
            "summary": data["summary_embeddings"],
            "reviewer_ids": data["reviewer_ids"],
            "model_name": str(data["model_name"]),
            "embedding_dim": int(data["embedding_dim"]),
        }
        metadata = pd.read_csv(metadata_file)
        return embeddings, metadata
    except Exception as e:
        st.warning(f"Failed to load embeddings/metadata: {e}")
        return None, None


def generate_paper_embeddings(model: SentenceTransformer, paper_data: dict) -> dict:
    title = paper_data.get("title") or "No title"
    abstract = paper_data.get("abstract") or "No abstract"
    keywords = ", ".join(paper_data.get("keywords", [])) or "No keywords"
    summary = paper_data.get("summary") or "No summary"
    return {
        "title": model.encode(title, convert_to_numpy=True),
        "abstract": model.encode(abstract, convert_to_numpy=True),
        "keywords": model.encode(keywords, convert_to_numpy=True),
        "summary": model.encode(summary, convert_to_numpy=True),
    }


def calculate_scores(paper_emb: dict, reviewer_emb: dict, weights: dict) -> list[dict]:
    n = len(reviewer_emb["reviewer_ids"])
    out = []
    for i in range(n):
        t = cosine_similarity(paper_emb["title"].reshape(1, -1), reviewer_emb["title"][i].reshape(1, -1))[0][0]
        a = cosine_similarity(paper_emb["abstract"].reshape(1, -1), reviewer_emb["abstract"][i].reshape(1, -1))[0][0]
        k = cosine_similarity(paper_emb["keywords"].reshape(1, -1), reviewer_emb["keywords"][i].reshape(1, -1))[0][0]
        s = cosine_similarity(paper_emb["summary"].reshape(1, -1), reviewer_emb["summary"][i].reshape(1, -1))[0][0]
        weighted = weights["abstract"]*a + weights["title"]*t + weights["keywords"]*k + weights["summary"]*s
        out.append({
            "reviewer_id": int(reviewer_emb["reviewer_ids"][i]),
            "weighted_score": float(weighted),
            "title_similarity": float(t),
            "abstract_similarity": float(a),
            "keywords_similarity": float(k),
            "summary_similarity": float(s),
        })
    return out


def top_k_reviewers(scores: list[dict], metadata: pd.DataFrame, k: int) -> pd.DataFrame:
    df = pd.DataFrame(scores).sort_values("weighted_score", ascending=False).head(k)
    merged = df.merge(metadata, on="reviewer_id", how="left")
    # Only keep Author and Score columns, rename weighted_score to Score
    result = merged[["Author", "weighted_score"]].copy()
    result.rename(columns={"weighted_score": "Score"}, inplace=True)
    return result


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Reviewer Recommendation System", layout="wide")
st.title("📄 Reviewer Recommendation System")
st.caption("Upload a research paper PDF to get reviewer recommendations.")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
top_k = st.slider("Number of reviewers", min_value=1, max_value=10, value=5)

run = st.button("Run Recommendation", type="primary", disabled=uploaded_pdf is None)

if uploaded_pdf is None:
    st.info("Please upload a PDF to begin.")
elif run:
    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    with st.spinner("Initializing…"):
        client = setup_groq_client()
        if client is None:
            st.stop()
        try:
            model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            st.error(f"Failed to load embedding model '{MODEL_NAME}': {e}")
            st.stop()

        reviewer_embeddings, reviewer_meta = load_reviewer_embeddings(EMBEDDINGS_FILE, METADATA_FILE)
        if reviewer_embeddings is None or reviewer_meta is None:
            st.stop()

        # Optional: check dim consistency
        try:
            emb_dim = int(reviewer_embeddings["embedding_dim"])
        except Exception:
            emb_dim = None

    with st.spinner("Reading PDF…"):
        text = extract_text_from_pdf(pdf_path)
        if not text:
            st.warning("Failed to extract text from the uploaded PDF.")
            st.stop()

    with st.spinner("Extracting paper metadata with LLM…"):
        paper_data = extract_paper_metadata(client, text)
        if paper_data is None:
            st.warning("Failed to extract metadata from the paper.")
            st.stop()

    with st.spinner("Generating embeddings…"):
        paper_emb = generate_paper_embeddings(model, paper_data)
        if emb_dim is not None and paper_emb["title"].shape[-1] != emb_dim:
            st.warning(
                f"Embedding dimension mismatch: paper={paper_emb['title'].shape[-1]} vs reviewers={emb_dim}. "
                "Ensure the same model was used to build reviewer embeddings."
            )

    with st.spinner("Scoring reviewers…"):
        try:
            scores = calculate_scores(paper_emb, reviewer_embeddings, WEIGHTS)
            top_df = top_k_reviewers(scores, reviewer_meta, k=top_k)
        except Exception as e:
            st.warning(f"Failed to compute similarity scores: {e}")
            st.stop()

    # --------- UI Output ---------
    st.subheader("📌 Extracted Paper Details")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Title:** {paper_data.get('title') or 'N/A'}")
        st.markdown(f"**Authors:** {', '.join(paper_data.get('authors') or []) or 'N/A'}")
        st.markdown(f"**Keywords:** {', '.join(paper_data.get('keywords') or []) or 'N/A'}")
        st.markdown("**Abstract:**")
        st.write((paper_data.get("abstract") or "N/A")[:800] + ("..." if paper_data.get("abstract") else ""))
        st.markdown("**Summary:**")
        st.write((paper_data.get("summary") or "N/A")[:800] + ("..." if paper_data.get("summary") else ""))
    with col2:
        st.metric("Candidates evaluated", len(reviewer_embeddings["reviewer_ids"]))
        st.metric("Top K", top_k)

    st.subheader("🏆 Top Recommended Reviewers")
    st.dataframe(top_df, use_container_width=True)

    # Optional: download button
    csv_bytes = top_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="top_reviewers.csv", mime="text/csv")
