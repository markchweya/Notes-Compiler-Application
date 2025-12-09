# app.py — Notes Compiler TURBO++ (Compile → Download) + Parallel Hybrid PDF OCR
#
# pip install streamlit openai pypdf python-docx python-pptx pillow pymupdf
#
# PowerShell:
#   $env:OPENAI_API_KEY="sk-..."
#   $env:STREAMLIT_SERVER_MAX_UPLOAD_SIZE="2000"
#   streamlit run app.py

import os
import io
import re
import time
import base64
import hashlib
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# ---- Optional deps
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from pptx import Presentation  # python-pptx
except Exception:
    Presentation = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Notes Compiler Turbo++", page_icon="⚡", layout="wide")
st.title("⚡ Notes Compiler (Turbo++) — Compile → Download")
st.caption("Faster: parallel OCR + parallel summarization + live progress + live per-file results. No Q&A.")

CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
.stButton>button, .stDownloadButton>button { border-radius: 14px; padding: 0.6rem 1rem; }
code { border-radius: 10px; padding: 0.15rem 0.35rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass
class DocItem:
    name: str
    kind: str         # 'pdf'|'docx'|'pptx'|'text'|'image'|'unknown'
    mime: str
    sha256: str
    text: Optional[str] = None
    image_bytes: Optional[bytes] = None
    raw_bytes: Optional[bytes] = None


# -----------------------------
# SPEED DEFAULTS
# -----------------------------
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.15

DEFAULT_CHUNK_TOKENS = 3600
DEFAULT_OVERLAP_TOKENS = 60

DEFAULT_BATCH_SIZE = 6
DEFAULT_MAX_WORKERS = 6

# OCR controls
PDF_PAGE_TEXT_MIN_CHARS = 80
PDF_OCR_MAX_PAGES = 0  # 0 = all
PDF_RENDER_ZOOM = 1.6
OCR_MAX_DIM = 1500
OCR_JPEG_QUALITY = 70

TOK_OCR_PAGE = 750
TOK_CHUNK_SUMMARY = 800
TOK_DOC_MERGE = 1400
TOK_GLOBAL_MERGE = 2000


# -----------------------------
# SESSION CACHE (in-session)
# -----------------------------
for k, v in [
    ("docs", {}),
    ("cache", {}),
    ("compiled", None),
    ("doc_notes", []),
    ("compiled_key", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


def cache_get(key: str):
    return st.session_state["cache"].get(key)


def cache_set(key: str, value: str):
    st.session_state["cache"][key] = value


# -----------------------------
# HELPERS
# -----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4)) if text else 0


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    text = clean_text(text)
    paras = split_paragraphs(text)
    if not paras:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_t = 0

    def flush():
        nonlocal cur, cur_t
        if cur:
            chunks.append("\n\n".join(cur).strip())
        cur, cur_t = [], 0

    for p in paras:
        pt = approx_tokens(p)
        if cur and cur_t + pt > target_tokens:
            flush()
            if overlap_tokens > 0 and chunks:
                tail = chunks[-1][-overlap_tokens * 4:].strip()
                if tail:
                    cur = [tail]
                    cur_t = approx_tokens(tail)
        cur.append(p)
        cur_t += pt

    flush()
    return [c for c in chunks if c.strip()]


def get_api_key() -> str:
    k = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    # strip accidental surrounding quotes
    if len(k) >= 2 and k[0] == k[-1] and k[0] in ("'", '"'):
        k = k[1:-1].strip()
    return k


_thread_local = threading.local()


def get_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("Missing openai SDK. Run: pip install openai")
    key = get_api_key()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY env var.")
    # thread-local client (safer + less overhead in parallel)
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=key)
    return _thread_local.client


def response_to_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    out = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        out.append(txt.strip())
    return "\n\n".join(out).strip()


def call_with_retry(fn, *, tries=6, base_sleep=0.6):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            if any(x in msg for x in ["429", "rate", "timeout", "temporarily", "overloaded", "502", "503", "504"]):
                time.sleep(base_sleep * (2 ** i))
                continue
            raise
    raise last


def oai_text(prompt: str, model: str, instructions: str, temperature: float, max_output_tokens: int, store: bool) -> str:
    client = get_client()

    def _do():
        return client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            store=store,
        )

    resp = call_with_retry(_do)
    return response_to_text(resp)


def oai_vision(prompt: str, image_bytes: bytes, mime: str, model: str, instructions: str,
              temperature: float, max_output_tokens: int, store: bool) -> str:
    client = get_client()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    input_items = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": data_url},
        ],
    }]

    def _do():
        return client.responses.create(
            model=model,
            instructions=instructions,
            input=input_items,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            store=store,
        )

    resp = call_with_retry(_do)
    return response_to_text(resp)


# -----------------------------
# FILE EXTRACTION
# -----------------------------
def extract_text_basic(name: str, data: bytes) -> Tuple[str, str]:
    ext = (name.split(".")[-1] or "").lower()

    if ext in ("txt", "md", "markdown", "csv", "log"):
        try:
            return clean_text(data.decode("utf-8")), "text"
        except Exception:
            return clean_text(data.decode("latin-1", errors="ignore")), "text"

    if ext == "pdf":
        if PdfReader is None:
            return "", "pdf"
        try:
            r = PdfReader(io.BytesIO(data))
            pages = []
            for p in r.pages:
                t = (p.extract_text() or "").strip()
                if t:
                    pages.append(t)
            return clean_text("\n\n".join(pages)), "pdf"
        except Exception:
            return "", "pdf"

    if ext == "docx":
        if docx is None:
            return "", "docx"
        try:
            d = docx.Document(io.BytesIO(data))
            return clean_text("\n".join([p.text for p in d.paragraphs if p.text.strip()])), "docx"
        except Exception:
            return "", "docx"

    if ext == "pptx":
        if Presentation is None:
            return "", "pptx"
        try:
            prs = Presentation(io.BytesIO(data))
            parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        t = shape.text.strip()
                        if t:
                            parts.append(t)
            return clean_text("\n".join(parts)), "pptx"
        except Exception:
            return "", "pptx"

    if ext in ("png", "jpg", "jpeg", "webp", "bmp"):
        return "", "image"

    return "", "unknown"


def maybe_downscale(img_bytes: bytes, max_dim: int) -> bytes:
    if Image is None:
        return img_bytes
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        if max(w, h) <= max_dim:
            return img_bytes
        scale = max_dim / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return img_bytes


def png_to_jpeg(png_bytes: bytes, quality: int) -> Tuple[bytes, str]:
    if Image is None:
        return png_bytes, "image/png"
    try:
        img = Image.open(io.BytesIO(png_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue(), "image/jpeg"
    except Exception:
        return png_bytes, "image/png"


# -----------------------------
# HYBRID PDF (text-first, OCR only where needed) — OCR pages in PARALLEL
# -----------------------------
def extract_pdf_hybrid_parallel(
    pdf_bytes: bytes,
    model: str,
    store: bool,
    ocr_workers: int,
    status_cb=None,
    progress_cb=None,
) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF missing. Install: pip install pymupdf")

    pdf_hash = sha256_bytes(pdf_bytes)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = doc.page_count
    if PDF_OCR_MAX_PAGES and PDF_OCR_MAX_PAGES > 0:
        n = min(n, int(PDF_OCR_MAX_PAGES))

    results: Dict[int, str] = {}
    ocr_jobs = []

    # Prepare OCR jobs (render in main thread; OCR in threads)
    for i in range(n):
        page_no = i + 1
        page = doc.load_page(i)

        t = (page.get_text("text") or "").strip()
        if len(t) >= PDF_PAGE_TEXT_MIN_CHARS:
            results[page_no] = t
            continue

        ck = f"pdfocr:{pdf_hash}:p{page_no}:z{PDF_RENDER_ZOOM}:md{OCR_MAX_DIM}:q{OCR_JPEG_QUALITY}:{model}"
        cached = cache_get(ck)
        if cached:
            results[page_no] = cached
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM), alpha=False)
        png = pix.tobytes("png")
        png = maybe_downscale(png, OCR_MAX_DIM)
        img_bytes, mime = png_to_jpeg(png, OCR_JPEG_QUALITY)

        ocr_jobs.append((page_no, ck, img_bytes, mime))

    total_ocr = len(ocr_jobs)
    if status_cb:
        status_cb(f"Pages: {n} | OCR needed: {total_ocr}")
    if progress_cb:
        progress_cb(0.0)

    def _ocr_one(job):
        page_no, ck, img_bytes, mime = job
        txt = oai_vision(
            "Extract ALL readable text from this page. Preserve headings/bullets/equations. Output text only.",
            img_bytes,
            mime,
            model=model,
            instructions="You are an OCR engine. Output text only. No commentary.",
            temperature=0.0,
            max_output_tokens=TOK_OCR_PAGE,
            store=store,
        )
        txt = clean_text(txt)
        return page_no, ck, txt

    done = 0
    if total_ocr > 0:
        with ThreadPoolExecutor(max_workers=max(1, int(ocr_workers))) as ex:
            futs = [ex.submit(_ocr_one, job) for job in ocr_jobs]
            for fut in as_completed(futs):
                page_no, ck, txt = fut.result()
                cache_set(ck, txt)
                results[page_no] = txt
                done += 1
                if status_cb:
                    status_cb(f"OCR progress: {done}/{total_ocr}")
                if progress_cb:
                    progress_cb(min(1.0, done / total_ocr))

    # join pages in order
    out = [results.get(p, "") for p in range(1, n + 1)]
    return clean_text("\n\n".join([x for x in out if x.strip()]))


# -----------------------------
# PROMPTS
# -----------------------------
BASE_INSTRUCTIONS = """You are a strict NOTES COMPILER.
- No fluff. Keep exam-relevant detail.
- Use Markdown headings + bullets.
- If unclear, write [CHECK] instead of guessing.
"""

def prompt_batch_summarize(doc_name: str, batch_chunks: List[str], level: str, mode: str) -> str:
    blocks = []
    for j, ch in enumerate(batch_chunks, 1):
        blocks.append(f"[CHUNK {j}]\n{ch}")
    joined = "\n\n".join(blocks)

    return f"""
Document: {doc_name}
DEPTH: {level}
MODE: {mode}

Summarize EACH chunk into compact study notes.

OUTPUT FORMAT (MUST FOLLOW EXACTLY):
<<<1>>>
...notes for chunk 1...
<<<2>>>
...notes for chunk 2...
(and so on)

Chunks:
{joined}
"""

def prompt_doc_merge(doc_name: str, summaries: List[str], level: str, mode: str) -> str:
    joined = "\n\n---\n\n".join(summaries)
    return f"""
Merge these chunk-notes into ONE clean note.

Document: {doc_name}
DEPTH: {level}
MODE: {mode}

Requirements:
- Mini Table of Contents
- Remove duplicates
- Consistent terminology
- Add: Quick Revision (10 bullets max) at top

Chunk-notes:
{joined}
"""

def prompt_global_merge(doc_notes: List[Tuple[str, str]], level: str, mode: str) -> str:
    body = "\n\n".join([f"## Source: {name}\n\n{note}" for name, note in doc_notes])
    return f"""
Compile ALL source notes into ONE master note.

DEPTH: {level}
MODE: {mode}

Requirements:
- Title + Table of Contents
- Group by topic (merge overlaps across files)
- End with: 1-page Cheat Sheet (dense bullets)

Source notes:
{body}
"""

def parse_batched_output(text: str, expected: int) -> List[str]:
    parts: Dict[int, str] = {}
    pattern = r"<<<\s*(\d+)\s*>>>\s*"
    splits = re.split(pattern, text.strip())
    if len(splits) < 3:
        return []
    for k in range(1, len(splits) - 1, 2):
        try:
            idx = int(splits[k])
        except Exception:
            continue
        body = splits[k + 1].strip()
        if body:
            parts[idx] = body
    out = [parts.get(i, "").strip() for i in range(1, expected + 1)]
    if any(not x for x in out):
        return []
    return out


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Speed Settings")

    k = get_api_key()
    if k:
        st.success(f"OPENAI_API_KEY detected ✅ (len={len(k)}, prefix={k[:6]})")
    else:
        st.warning("Set OPENAI_API_KEY in your terminal.")

    model = st.text_input("Model", value=DEFAULT_MODEL)
    store = st.toggle("Store responses on OpenAI", value=False)
    temperature = st.slider("Temperature", 0.0, 1.0, float(DEFAULT_TEMPERATURE), 0.05)

    st.divider()
    level = st.selectbox("Depth", ["Simple (high school)", "Standard (A-level)", "Advanced (university)"], index=1)
    mode = st.selectbox("Mode", ["Study Notes", "Exam Revision", "Lecture Cleanup", "Flashcards-first"], index=1)

    st.divider()
    chunk_tokens = st.slider("Chunk size (tokens)", 1200, 5500, DEFAULT_CHUNK_TOKENS, 100)
    overlap_tokens = st.slider("Chunk overlap", 0, 400, DEFAULT_OVERLAP_TOKENS, 10)

    st.divider()
    batch_size = st.slider("Chunks per API call (batch)", 2, 10, DEFAULT_BATCH_SIZE, 1)
    max_workers = st.slider("Parallel workers (summaries)", 1, 12, DEFAULT_MAX_WORKERS, 1)
    ocr_workers = st.slider("Parallel workers (OCR pages)", 1, 12, max(2, DEFAULT_MAX_WORKERS // 2), 1)

    st.divider()
    st.subheader("🧾 PDFs")
    if fitz is None:
        st.info("Install for scanned PDFs: `pip install pymupdf`")
    pdf_min_chars = st.slider("Page text threshold (skip OCR if >= chars)", 0, 400, PDF_PAGE_TEXT_MIN_CHARS, 10)
    max_pages = st.number_input("Max PDF pages to process (0=all)", 0, 10000, PDF_OCR_MAX_PAGES, 5)

PDF_PAGE_TEXT_MIN_CHARS = int(pdf_min_chars)
PDF_OCR_MAX_PAGES = int(max_pages)


# -----------------------------
# UPLOAD + CONTROLS
# -----------------------------
uploaded = st.file_uploader(
    "Upload notes (PDF/DOCX/PPTX/TXT/MD + images). Multiple files supported.",
    type=["pdf", "docx", "pptx", "txt", "md", "markdown", "png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)

c1, c2 = st.columns([1, 1])
compile_btn = c1.button("🚀 Compile", type="primary", use_container_width=True)
clear_btn = c2.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state["docs"] = {}
    st.session_state["cache"] = {}
    st.session_state["compiled"] = None
    st.session_state["doc_notes"] = []
    st.session_state["compiled_key"] = None
    st.toast("Cleared ✅")
    st.stop()


def ingest(files) -> List[DocItem]:
    items: List[DocItem] = []
    for f in files or []:
        data = f.getvalue()
        sha = sha256_bytes(data)
        if sha in st.session_state["docs"]:
            items.append(st.session_state["docs"][sha])
            continue

        txt, kind = extract_text_basic(f.name, data)
        mime = getattr(f, "type", "") or "application/octet-stream"

        it = DocItem(
            name=f.name,
            kind=kind,
            mime=mime,
            sha256=sha,
            text=txt if txt else None,
            image_bytes=data if kind == "image" else None,
            raw_bytes=data if kind == "pdf" else None,
        )
        st.session_state["docs"][sha] = it
        items.append(it)
    return items


docs = ingest(uploaded)
if docs:
    st.info(f"Loaded {len(docs)} file(s).")


# -----------------------------
# PIPELINE
# -----------------------------
def compile_all(docs: List[DocItem]) -> str:
    if not docs:
        raise RuntimeError("No files uploaded.")
    if OpenAI is None:
        raise RuntimeError("Missing openai SDK. Install: pip install openai")

    t0 = time.time()

    # 1) OCR images
    img_docs = [d for d in docs if d.kind == "image"]
    if img_docs:
        st.subheader("🖼️ Extracting text from images")
        for d in img_docs:
            ck = f"imgocr:{d.sha256}:{model}"
            cached = cache_get(ck)
            if cached:
                d.text = cached
                continue
            txt = oai_vision(
                "Extract ALL readable text. Output text only.",
                d.image_bytes or b"",
                d.mime or "image/png",
                model=model,
                instructions="You are an OCR engine. Output text only.",
                temperature=0.0,
                max_output_tokens=TOK_OCR_PAGE,
                store=store,
            )
            d.text = clean_text(txt)
            cache_set(ck, d.text)

    # 2) PDFs: hybrid extraction (text-first; OCR pages in parallel)
    pdfs = [d for d in docs if d.kind == "pdf"]
    if pdfs:
        st.subheader("📄 Processing PDFs (hybrid: text-first, OCR only pages that need it)")
        if fitz is None:
            # If any PDF has little extracted text, we can't OCR without pymupdf
            low = [d for d in pdfs if not (d.text or "").strip()]
            if low:
                raise RuntimeError("Some PDFs need OCR but PyMuPDF is missing. Install: pip install pymupdf")
        pdf_container = st.container()
        for d in pdfs:
            box = pdf_container.container()
            box.write(f"**{d.name}**")
            msg = box.empty()
            pb = box.progress(0.0)

            if d.raw_bytes is None:
                msg.write("Skipping (no bytes).")
                continue

            # Always run hybrid if pymupdf exists (it will skip OCR for text pages)
            if fitz is not None:
                d.text = extract_pdf_hybrid_parallel(
                    d.raw_bytes,
                    model=model,
                    store=store,
                    ocr_workers=int(ocr_workers),
                    status_cb=lambda m, _msg=msg: _msg.write(m),
                    progress_cb=lambda p, _pb=pb: _pb.progress(min(1.0, p)),
                )
                pb.progress(1.0)
            else:
                # fallback: keep pypdf text already extracted
                msg.write("PyMuPDF not installed: using embedded text only.")
                pb.progress(1.0)

    # 3) Chunking
    usable = [d for d in docs if (d.text or "").strip()]
    if not usable:
        raise RuntimeError("No usable text extracted.")

    st.subheader("🧩 Chunking + batch summarization (parallel)")
    progress = st.progress(0.0)
    status = st.empty()

    tasks = []
    for d in usable:
        chunks = chunk_text(d.text or "", int(chunk_tokens), int(overlap_tokens))
        for b0 in range(0, len(chunks), int(batch_size)):
            batch = chunks[b0:b0 + int(batch_size)]
            tasks.append((d, b0 // int(batch_size), batch))

    total = max(1, len(tasks))
    done = 0

    per_doc_summaries: Dict[str, Dict[int, List[str]]] = {d.sha256: {} for d in usable}

    def run_batch(d: DocItem, batch_idx: int, batch_chunks: List[str]) -> Tuple[str, int, List[str]]:
        ck = f"sum:{d.sha256}:b{batch_idx}:m{model}:lv{level}:md{mode}:ct{chunk_tokens}:ol{overlap_tokens}:bs{batch_size}"
        cached = cache_get(ck)
        if cached:
            parsed = parse_batched_output(cached, expected=len(batch_chunks))
            if parsed:
                return d.sha256, batch_idx, parsed

        out = oai_text(
            prompt_batch_summarize(d.name, batch_chunks, level, mode),
            model=model,
            instructions=BASE_INSTRUCTIONS,
            temperature=temperature,
            max_output_tokens=TOK_CHUNK_SUMMARY * len(batch_chunks),
            store=store,
        )
        out = clean_text(out)
        cache_set(ck, out)

        parsed = parse_batched_output(out, expected=len(batch_chunks))
        if not parsed:
            parsed = [out] * len(batch_chunks)
        return d.sha256, batch_idx, parsed

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futs = [ex.submit(run_batch, d, bi, bch) for (d, bi, bch) in tasks]
        for fut in as_completed(futs):
            doc_sha, batch_idx, summaries = fut.result()
            per_doc_summaries[doc_sha][batch_idx] = summaries
            done += 1
            status.write(f"Summarized batches: {done}/{total}")
            progress.progress(min(1.0, done / total))

    # 4) Merge each document (parallel) + live output
    st.subheader("📎 Merging each document (live)")
    live_area = st.container()
    placeholders = {d.sha256: live_area.empty() for d in usable}

    def merge_one(d: DocItem) -> Tuple[str, str, str]:
        batches = per_doc_summaries[d.sha256]
        all_summaries = []
        for i in sorted(batches.keys()):
            all_summaries.extend(batches[i])

        mk = f"docmerge:{d.sha256}:m{model}:lv{level}:md{mode}"
        cached = cache_get(mk)
        if cached:
            merged = cached
        else:
            merged = oai_text(
                prompt_doc_merge(d.name, all_summaries, level, mode),
                model=model,
                instructions=BASE_INSTRUCTIONS,
                temperature=min(0.25, temperature),
                max_output_tokens=TOK_DOC_MERGE,
                store=store,
            )
            merged = clean_text(merged)
            cache_set(mk, merged)

        return d.sha256, d.name, merged

    merged_map: Dict[str, Tuple[str, str]] = {}

    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), len(usable)))) as ex:
        mfuts = [ex.submit(merge_one, d) for d in usable]
        for fut in as_completed(mfuts):
            sha, name, merged = fut.result()
            merged_map[sha] = (name, merged)
            placeholders[sha].markdown(f"### ✅ Finished: {name}\n\n{merged}")

    # keep consistent ordering by upload order
    doc_notes = [(d.name, merged_map[d.sha256][1]) for d in usable]
    st.session_state["doc_notes"] = doc_notes

    # 5) Global compile
    st.subheader("📘 Building master notes")
    compiled_key = f"{model}|{level}|{mode}|" + "|".join(sorted([d.sha256 for d in usable]))
    if st.session_state.get("compiled") and st.session_state.get("compiled_key") == compiled_key:
        return st.session_state["compiled"]

    compiled = oai_text(
        prompt_global_merge(doc_notes, level, mode),
        model=model,
        instructions=BASE_INSTRUCTIONS,
        temperature=min(0.25, temperature),
        max_output_tokens=TOK_GLOBAL_MERGE,
        store=store,
    )
    compiled = clean_text(compiled)

    st.session_state["compiled_key"] = compiled_key
    st.session_state["compiled"] = compiled

    st.caption(f"Total time: {time.time() - t0:.1f}s")
    return compiled


# -----------------------------
# RUN
# -----------------------------
if compile_btn:
    try:
        _ = compile_all(docs)
        st.success("Done ✅ Scroll to download.")
    except Exception as e:
        st.error(f"Compile failed: {e}")


# -----------------------------
# OUTPUT
# -----------------------------
compiled = st.session_state.get("compiled")
if compiled:
    st.subheader("✅ Compiled Notes (Preview)")
    st.markdown(compiled)

    data = compiled.encode("utf-8")
    a, b = st.columns([1, 1])
    a.download_button("⬇️ Download .md", data=data, file_name="compiled_notes.md", mime="text/markdown")
    b.download_button("⬇️ Download .txt", data=data, file_name="compiled_notes.txt", mime="text/plain")

    with st.expander("Per-file merged notes"):
        for name, note in st.session_state.get("doc_notes", []):
            st.markdown(f"### {name}")
            st.markdown(note)
            st.divider()
