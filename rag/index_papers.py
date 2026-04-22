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
        text = pdf_to_text(pdf)
        parts = simple_chunk(text, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
        sid = source_id_for_file(pdf)
        title = pdf.stem
        for i, chunk in enumerate(parts):
            all_chunks.append(Chunk(source_id=sid, title=title, year=None, url=None, chunk_index=i, content=chunk))

    # Embed + upload in batches
    for i in tqdm(range(0, len(all_chunks), args.batch), desc="Embedding+upload"):
        batch_chunks = all_chunks[i : i + args.batch]
        embs = embed_texts(client, [c.content for c in batch_chunks], model=args.embed_model)
        upsert_chunks(sb, table=args.table, chunks=batch_chunks, embeddings=embs)

    print(f"Done. Uploaded {len(all_chunks)} chunks to {args.table}.")


if __name__ == "__main__":
    main()

