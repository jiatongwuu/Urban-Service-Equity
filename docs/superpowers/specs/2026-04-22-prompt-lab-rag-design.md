## Goal

Update the static “Prompt Lab” page (`docs/chat.html` + `docs/assets/chat.js`) so it uses the repo’s deployed RAG backend (`/api/chat`) and can retrieve knowledge from the indexed papers (Supabase pgvector), without requiring any browser-entered LLM API keys.

## Non-goals

- Building a general production chat UI or auth system.
- Adding a full “Sources” panel UI. Citations remain inline (e.g. `[ref:1]`), as produced by the backend.
- Supporting dual-mode (direct-to-provider) requests from the browser.

## Current state (context)

- Prompt Lab is a static UI (`docs/chat.html`) driven by `docs/assets/chat.js`.
  - It currently calls OpenAI/Anthropic/Google directly from the browser using a user-entered API key.
  - It also does a small, local-only excerpt retrieval for one bundled paper (`docs/sociopaper/zahnow1.txt`) by token matching; this is separate from the Supabase RAG pipeline.
- The repo already has a deployable RAG API: `api/chat.ts`.
  - It embeds the query, retrieves top-k chunks via Supabase RPC `match_paper_chunks`, and returns `{ content, citations }`.
  - It constructs an augmented system message that instructs inline citations in the form `[ref:1]`, `[ref:2]`, etc.

## Selected approach

### Approach A: Server-backed Prompt Lab (chosen)

Prompt Lab becomes a thin client that always posts to a deployed backend endpoint:

- Browser never calls OpenAI/Anthropic/Google directly.
- Browser never asks for or stores API keys.
- Prompt Lab still provides the prompt-engineering controls (system prompt, temperature, max tokens, dashboard context mode), but execution is via the backend.

## UX / UI design

### Remove

- “API key (dev only)” input and any related localStorage persistence.
- Provider-specific direct-call logic (OpenAI/Anthropic/Google HTTP calls).

### Keep

- Model selector + optional “Custom model ID” override.
- Temperature and max tokens controls.
- System prompt template textbox.
- Context mode controls (global / cluster / grid) and “resolved context payload” preview.
- Existing chat history UX and max history bounding.

### Add (minimal)

- RAG API base configuration:
  - Default to the current deployment base used in `docs/assets/rag_client.js`.
  - Stored in localStorage so GitHub Pages and local dev can target different backends.
- RAG top-k control (optional but recommended):
  - Defaults to 8 (matching current backend default).

### Citations display

- No separate sources panel.
- The assistant message is rendered exactly as returned by the backend, including inline `[ref:n]` markers.

## Data flow

### On send (frontend)

1. Compute the resolved system prompt string (same strategy as current Prompt Lab):
   - User-edited system prompt template
   - Output-style guard rails
   - Dashboard context payload JSON (global / cluster / grid)
   - (Remove the local `zahnow1` excerpt block; paper knowledge now comes from backend RAG.)
2. Build request payload and POST to the backend:

```json
{
  "messages": [{ "role": "user|assistant", "content": "..." }, "..."],
  "question": "latest user message (optional redundancy)",
  "systemPrompt": "resolved system prompt + dashboard context payload",
  "model": "selected model id",
  "temperature": 0.3,
  "maxTokens": 900,
  "topK": 8
}
```

3. Render `data.content` as the assistant message.
4. Keep storing the conversation state in localStorage (bounded by `MAX_HISTORY`) for convenience.

### Backend (`api/chat.ts`)

- Validate inputs and bound:
  - `topK` within a reasonable range (e.g. 1–20).
  - `maxTokens` within a reasonable range (to protect cost and latency).
- Use the provided `systemPrompt` as the base system instruction and append:
  - Retrieved “Paper excerpts” block
  - Inline citation instruction (“Cite sources like [ref:1]…”)
- Return:
  - `content` (assistant text)
  - `citations` (structured list, currently unused in UI but retained for future)

## Deployment / integration notes

- Prompt Lab must work on GitHub Pages against a deployed Vercel backend.
  - `api/chat.ts` already sets permissive CORS for `POST` and `OPTIONS`.
- Environment variables remain server-side (Vercel):
  - `OPENAI_API_KEY`
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY` (preferred) or `SUPABASE_ANON_KEY`

## Error handling

- Frontend:
  - Show a clear status if the backend base is missing/misconfigured.
  - Surface backend errors (HTTP status + message) in the chat stream as an assistant “Error:” message (matching existing UX pattern).
- Backend:
  - Keep current structured `{ error: "..." }` responses.

## Testing / verification

- Local UI smoke test:
  - Serve `docs/` locally and verify Prompt Lab can send a message and receive a response via the configured RAG backend base.
- Deployed smoke test:
  - On GitHub Pages, set the backend base to the Vercel deployment and verify end-to-end retrieval and inline `[ref:n]` citations appear for paper-grounded questions.

