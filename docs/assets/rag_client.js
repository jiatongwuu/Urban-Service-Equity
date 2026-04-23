// Minimal client helper for the static site to call a deployed RAG backend.
// Usage: set window.RAG_API_BASE = "https://<vercel-app>.vercel.app"
// (Do not overwrite if already set by the page.)
window.RAG_API_BASE = window.RAG_API_BASE || "https://urban-service-equity.vercel.app";

export async function ragChat({ question, messages = [], systemPrompt = "", model = "gpt-4o", topK = 8 }) {
  const base = (window.RAG_API_BASE || "").replace(/\/$/, "");
  if (!base) throw new Error("Missing window.RAG_API_BASE");
  const r = await fetch(`${base}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, messages, systemPrompt, model, topK }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data?.error || `HTTP ${r.status}`);
  return data;
}

