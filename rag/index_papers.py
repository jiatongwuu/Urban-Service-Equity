from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from supabase import create_client
from tqdm import tqdm


@dataclass
class Chunk:
    source_id: str
    title: str
    year: Optional[int]
    url: Optional[str]
    chunk_index: int
    content: str
    content_tokens: Optional[int] = None


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _first_page_text(reader: PdfReader) -> str:
    try:
        if not reader.pages:
            return ""
        return _clean_text(reader.pages[0].extract_text() or "")
    except Exception:
        return ""


def _guess_title(pdf: Path, reader: PdfReader, first_page: str) -> str:
    # Prefer metadata title when it looks meaningful, otherwise fall back to filename stem.
    meta_title = None
    try:
        meta = getattr(reader, "metadata", None)
        meta_title = getattr(meta, "title", None) if meta else None
    except Exception:
        meta_title = None
    if meta_title:
        t = str(meta_title).strip()
        if t and t.lower() not in {"untitled", "title"} and len(t) >= 8:
            return t

    # Heuristic: use first non-empty line from first page if it looks like a title.
    lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
    if lines:
        cand = lines[0]
        # If the first line is very short (e.g. a running header), try a couple more lines.
        if len(cand) < 12 and len(lines) >= 3:
            cand = " ".join(lines[:2]).strip()
        if 12 <= len(cand) <= 200:
            return cand

    return pdf.stem


def _guess_year(pdf: Path, text: str) -> Optional[int]:
    # Look for a plausible year near the front matter.
    m = re.search(r"\b(19\d{2}|20[0-2]\d)\b", text[:3000])
    if not m:
        return None
    y = int(m.group(1))
    return y


def _guess_authors(first_page: str, title: str) -> Optional[str]:
    """
    Best-effort author line extraction.
    We avoid trying to be perfect; goal is to make common papers answerable (\"who are the authors\").
    """
    if not first_page:
        return None
    txt = first_page
    # Remove the title if it appears verbatim, to reduce false positives.
    t = re.sub(r"\s+", " ", title).strip().lower()
    norm = re.sub(r"\s+", " ", txt).strip()
    norm_lower = norm.lower()
    if t and t in norm_lower:
        # keep everything after the title occurrence
        idx = norm_lower.find(t)
        norm = norm[idx + len(t) :].strip()

    # Stop at abstract/introduction keywords.
    stop = re.split(r"\bAbstract\b|\bABSTRACT\b|\bIntroduction\b|\b1\s+Introduction\b", norm, maxsplit=1)
    head = stop[0]
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]

    # Filter out obvious non-author lines.
    bad = re.compile(r"(university|department|institute|email|@|http|doi|arxiv|copyright|accepted|published)", re.I)
    cleaned = [ln for ln in lines[:12] if not bad.search(ln)]
    if not cleaned:
        return None

    # Authors are often in the first 1-3 lines after title.
    # Prefer a line with commas / 'and' / superscripts.
    score = []
    for ln in cleaned[:6]:
        s = 0
        if "," in ln:
            s += 2
        if re.search(r"\band\b", ln, re.I):
            s += 1
        if re.search(r"[A-Z][a-z]+", ln):
            s += 1
        if len(ln) <= 160:
            s += 1
        score.append((s, ln))
    score.sort(reverse=True, key=lambda x: x[0])
    best = score[0][1] if score else None
    if not best or len(best) < 8:
        return None
    return best


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return _clean_text("\n".join(parts))


def simple_chunk(text: str, *, max_chars: int = 2200, overlap_chars: int = 250) -> List[str]:
    # Char-based chunking avoids tokenizer dependency and is good enough for initial RAG.
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(text):
            break
        i = max(0, j - overlap_chars)
    return chunks


def source_id_for_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.name.encode("utf-8"))
    try:
        h.update(str(path.stat().st_size).encode("utf-8"))
        h.update(str(int(path.stat().st_mtime)).encode("utf-8"))
    except Exception:
        pass
    return h.hexdigest()[:16]


def embed_texts(client: OpenAI, texts: List[str], *, model: str) -> List[List[float]]:
    res = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in res.data]


def upsert_chunks(
    sb,
    *,
    table: str,
    chunks: List[Chunk],
    embeddings: List[List[float]],
) -> None:
    rows = []
    for ch, emb in zip(chunks, embeddings):
        rows.append(
            {
                "source_id": ch.source_id,
                "source_title": ch.title,
                "source_year": ch.year,
                "source_url": ch.url,
                "chunk_index": ch.chunk_index,
                "content": ch.content,
                "content_tokens": ch.content_tokens,
                "embedding": emb,
            }
        )
    # Supabase Python client uses upsert() with on_conflict; simplest is insert for now
    # and rely on re-indexing into a fresh table. If you want true upsert, add a unique
    # constraint (source_id, chunk_index).
    sb.table(table).insert(rows).execute()


def delete_source_rows(sb, *, table: str, source_id: str) -> None:
    # Make re-indexing idempotent per PDF without requiring a DB migration.
    sb.table(table).delete().eq("source_id", source_id).execute()


def iter_pdfs(papers_dir: Path) -> Iterable[Path]:
    for p in papers_dir.rglob("*.pdf"):
        if p.name.startswith("~$"):
            continue
        yield p


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Index paper PDFs into Supabase pgvector")
    ap.add_argument("--papers-dir", required=True, help="Folder containing PDF papers (local or Drive-mounted)")
    ap.add_argument("--table", default=os.environ.get("SUPABASE_TABLE", "paper_chunks"))
    ap.add_argument("--embed-model", default=os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--overlap-chars", type=int, default=250)
    ap.add_argument(
        "--no-delete-existing",
        action="store_true",
        help="Do not delete existing rows for a source_id before inserting (may create duplicates).",
    )
    args = ap.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    if not supabase_url or not supabase_service_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    client = OpenAI(api_key=openai_key)
    sb = create_client(supabase_url, supabase_service_key)

    papers_dir = Path(args.papers_dir).expanduser().resolve()
    pdfs = list(iter_pdfs(papers_dir))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {papers_dir}")

    all_chunks: List[Chunk] = []
    for pdf in tqdm(pdfs, desc="Reading PDFs"):
        reader = PdfReader(str(pdf))
        first_page = _first_page_text(reader)
        text = _clean_text("\n".join([(p.extract_text() or "") if p is not None else "" for p in reader.pages]))
        parts = simple_chunk(text, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
        sid = source_id_for_file(pdf)
        title = _guess_title(pdf, reader, first_page)
        year = _guess_year(pdf, first_page) or _guess_year(pdf, text)
        authors = _guess_authors(first_page, title)

        # Insert a dedicated "header chunk" so author/title questions retrieve reliably.
        header_lines = [f"Title: {title}"]
        if authors:
            header_lines.append(f"Authors: {authors}")
        if year:
            header_lines.append(f"Year: {year}")
        header_lines.append(f"Filename: {pdf.name}")
        header_lines.append("")
        header_lines.append("First page excerpt:")
        header_lines.append(first_page[:1800] if first_page else "(no text extracted from first page)")
        header = "\n".join(header_lines).strip()
        all_chunks.append(Chunk(source_id=sid, title=title, year=year, url=None, chunk_index=0, content=header))

        # Shift indices so (source_id, chunk_index) stays stable across re-runs.
        for i, chunk in enumerate(parts):
            all_chunks.append(Chunk(source_id=sid, title=title, year=year, url=None, chunk_index=i + 1, content=chunk))

    # Embed + upload in batches
    if not args.no_delete_existing:
        for pdf in tqdm(pdfs, desc="Deleting existing rows"):
            delete_source_rows(sb, table=args.table, source_id=source_id_for_file(pdf))

    for i in tqdm(range(0, len(all_chunks), args.batch), desc="Embedding+upload"):
        batch_chunks = all_chunks[i : i + args.batch]
        embs = embed_texts(client, [c.content for c in batch_chunks], model=args.embed_model)
        upsert_chunks(sb, table=args.table, chunks=batch_chunks, embeddings=embs)

    print(f"Done. Uploaded {len(all_chunks)} chunks to {args.table}.")


if __name__ == "__main__":
    main()

