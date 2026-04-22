-- Enable pgvector
create extension if not exists vector;

-- Store chunked paper text + embeddings
create table if not exists public.paper_chunks (
  id bigserial primary key,
  source_id text not null,
  source_title text,
  source_year int,
  source_url text,
  chunk_index int not null,
  content text not null,
  content_tokens int,
  embedding vector(1536), -- matches text-embedding-3-small output dim
  created_at timestamptz default now()
);

-- Recommended uniqueness if you re-index incrementally
alter table public.paper_chunks
  add constraint if not exists paper_chunks_source_chunk_unique unique (source_id, chunk_index);

create index if not exists paper_chunks_source_id_idx on public.paper_chunks (source_id);
create index if not exists paper_chunks_embedding_idx on public.paper_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Similarity search RPC (Supabase client friendly)
create or replace function public.match_paper_chunks(
  query_embedding vector(1536),
  match_count int default 8
)
returns table (
  id bigint,
  source_id text,
  source_title text,
  source_year int,
  source_url text,
  chunk_index int,
  content text,
  similarity float
)
language sql stable
as $$
  select
    pc.id,
    pc.source_id,
    pc.source_title,
    pc.source_year,
    pc.source_url,
    pc.chunk_index,
    pc.content,
    1 - (pc.embedding <=> query_embedding) as similarity
  from public.paper_chunks pc
  where pc.embedding is not null
  order by pc.embedding <=> query_embedding
  limit match_count;
$$;

-- If RLS is enabled, anon clients (like serverless functions using anon key) may see 0 rows.
-- Option A (recommended for backend retrieval): use service role key in the server and keep RLS strict.
-- Option B (quick dev): allow read access for retrieval via anon key.
alter table public.paper_chunks enable row level security;

drop policy if exists "paper_chunks_read_anon" on public.paper_chunks;
create policy "paper_chunks_read_anon"
on public.paper_chunks
for select
to anon
using (true);


