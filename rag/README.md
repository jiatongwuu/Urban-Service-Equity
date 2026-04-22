# RAG setup (papers → citations in chat)

This repo’s dashboard is static (`docs/`). To add papers as sources, we use a **separate RAG backend**:

- **Corpus**: PDFs in Google Drive (or local folder)
- **Index**: embeddings stored in **Supabase pgvector**
- **API**: deploy `/api/chat` (Vercel) and call it from the static site

## 1) Put papers in a folder

Create a folder (local or Drive-mounted) like:

- `data/papers/` (local)
- `/content/drive/My Drive/<team>/data/papers/` (Colab Drive mount)

PDF file names can be anything.

## 2) Create Supabase vector table + search function

Run the SQL in `rag/supabase_schema.sql` in your Supabase project SQL editor.

You’ll need these env vars later:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_TABLE` (defaults to `paper_chunks`)

## 3) Index papers (PDF → chunks → embeddings → Supabase)

```bash
python -m pip install -r requirements-rag.txt
export OPENAI_API_KEY="..."
export SUPABASE_URL="..."
export SUPABASE_SERVICE_ROLE_KEY="..."
python rag/index_papers.py --papers-dir "/path/to/papers"
```

This can be run on any machine that can read the PDFs (including Colab with Drive mounted).

## 4) Deploy the chat API (Vercel)

This repo includes Vercel serverless functions under `api/`.

Set Vercel env vars:

- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (recommended; avoids RLS issues)
- `SUPABASE_ANON_KEY` (optional fallback)

Then deploy. The API endpoint will be:

- `https://<your-vercel-app>.vercel.app/api/chat`

## 5) Point the static site at the API

On GitHub Pages (static site), set:

- `window.RAG_API_BASE = "https://<your-vercel-app>.vercel.app"`

See `docs/assets/rag_client.js`.

