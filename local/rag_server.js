/**
 * Local RAG backend for the static website chatbox.
 *
 * Runs: POST http://localhost:8787/api/chat
 * Request body: { question, messages?, systemPrompt?, model?, topK?, temperature?, maxTokens?, debug? }
 * Response: { content, citations, retrievalDebug? }
 *
 * No external deps (so it's easy to run). Reads env vars from process.env and
 * also best-effort loads .env.local from repo root.
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const OpenAI = require("openai").default;
const { createClient } = require("@supabase/supabase-js");

function loadDotEnvLocal() {
  try {
    const p = path.resolve(__dirname, "..", ".env.local");
    if (!fs.existsSync(p)) return;
    const raw = fs.readFileSync(p, "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const s = line.trim();
      if (!s || s.startsWith("#")) continue;
      const m = s.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
      if (!m) continue;
      const k = m[1];
      let v = m[2] || "";
      // Strip inline comments (very simple heuristic)
      if (!v.startsWith('"') && !v.startsWith("'")) v = v.split(" #")[0].trim();
      v = v.replace(/^"(.*)"$/, "$1").replace(/^'(.*)'$/, "$1");
      if (process.env[k] == null) process.env[k] = v;
    }
  } catch {
    // ignore
  }
}

function json(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type",
  });
  res.end(body);
}

function extractQuotedTitle(q) {
  const s = String(q || "");
  const m = s.match(/[“"'‘]([^”"'’]{18,260})[”"'’]/);
  if (!m) return null;
  const title = m[1].trim();
  if (title.split(/\s+/).length < 4) return null;
  return title;
}

function extractParenTitle(q) {
  const s = String(q || "");
  const m = s.match(/([A-Za-z0-9][\s\S]{18,260}?)\s*\(\s*(19\d{2}|20\d{2})\s*\)/);
  if (!m) return null;
  const title = m[1].trim();
  if (title.split(/\s+/).length < 4) return null;
  return title;
}

function wantsAuthorOrAbstract(q) {
  const s = String(q || "").toLowerCase();
  return /\b(author|authors|who wrote|written by|abstract)\b/.test(s);
}

function findTitleHintFromHistory(q, messages) {
  const direct = extractQuotedTitle(q) || extractParenTitle(q);
  if (direct) return direct;
  for (let i = (messages?.length || 0) - 1; i >= 0; i--) {
    const c = String(messages[i]?.content || "");
    const hint = extractQuotedTitle(c) || extractParenTitle(c);
    if (hint) return hint;
  }
  return null;
}

async function readJsonBody(req) {
  return await new Promise((resolve, reject) => {
    let buf = "";
    req.on("data", (d) => (buf += d));
    req.on("end", () => {
      try {
        resolve(buf ? JSON.parse(buf) : {});
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function buildRagRules() {
  return [
    "You are a helpful assistant with access to the section 'Paper excerpts' below.",
    "",
    "Hard rules:",
    "- Do NOT say you 'don't have access' or 'can't access the paper' when excerpts are present. You DO have access to those excerpts.",
    "- Do NOT deflect with scope refusals. Answer the user's question directly.",
    "- If Paper excerpts says 'No paper excerpts were retrieved.', say you couldn't find anything in the RAG index for this request and give concrete troubleshooting steps (confirm backend URL, confirm Supabase project/key, confirm chunks exist with embeddings). Do NOT claim you read the paper.",
    "",
    "How to answer using excerpts:",
    "- If the user asks about a specific paper (title/authors/year/venue/claims) and the relevant info appears in the excerpts, QUOTE the exact line(s) and cite them, e.g. [ref:1].",
    "- If the user asks for the full paper text, do NOT reproduce the full paper. Instead: say you can't provide full text, then summarize what the excerpts say and include 1–3 short quoted snippets with citations.",
    "- If excerpts do not contain the requested detail (e.g., author line missing), say that plainly and suggest what you *can* do.",
    "",
    "Citations:",
    "- Put citations right after the sentence they support, like [ref:2].",
    "- Never invent citations.",
  ].join("\n");
}

loadDotEnvLocal();

const PORT = Number(process.env.RAG_LOCAL_PORT || 8787);

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") return json(res, 204, {});
  if (req.url !== "/api/chat") return json(res, 404, { error: "Not found" });
  if (req.method !== "POST") return json(res, 405, { error: "Method not allowed" });

  try {
    const body = await readJsonBody(req);

    const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
    const SUPABASE_URL = process.env.SUPABASE_URL;
    const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;
    const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
    if (!OPENAI_API_KEY) return json(res, 500, { error: "Missing OPENAI_API_KEY" });
    if (!SUPABASE_URL) return json(res, 500, { error: "Missing SUPABASE_URL" });
    if (!SUPABASE_ANON_KEY && !SUPABASE_SERVICE_ROLE_KEY) {
      return json(res, 500, { error: "Missing SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)" });
    }

    const question = String(body.question || "").trim();
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const systemPrompt = String(body.systemPrompt || "").trim();
    const model = String(body.model || "gpt-4o");
    const topK = Number(body.topK || 8);
    const debug = Boolean(body.debug);
    const temperature = Number(body.temperature ?? 0.3);
    const maxTokens = Number(body.maxTokens ?? 900);

    const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    const keyType = SUPABASE_SERVICE_ROLE_KEY ? "service_role" : "anon";
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY || SUPABASE_ANON_KEY);

    const q = question || String(messages[messages.length - 1]?.content || "");
    const titleHint = findTitleHintFromHistory(q, messages);
    const needHeader = wantsAuthorOrAbstract(q);

    const retrievalDebug = {
      supabaseUrlHost: (() => {
        try {
          return new URL(SUPABASE_URL).host;
        } catch {
          return undefined;
        }
      })(),
      keyType,
      titleHint,
      retrievalMode: "none",
      visibleRowCount: null,
    };

    try {
      const { count, error } = await supabase.from("paper_chunks").select("id", { count: "exact", head: true });
      if (!error) retrievalDebug.visibleRowCount = typeof count === "number" ? count : null;
    } catch {
      // ignore
    }

    let chunks = [];
    if (titleHint) {
      retrievalDebug.retrievalMode = "title";
      const { data: byTitle, error: titleErr } = await supabase
        .from("paper_chunks")
        .select("source_id, source_title, source_year, source_url, chunk_index, content")
        .ilike("source_title", `%${titleHint}%`)
        .order("chunk_index", { ascending: true })
        .limit(Math.max(12, Math.min(40, topK * 2)));
      if (titleErr) return json(res, 500, { error: `Supabase title lookup error: ${titleErr.message}` });
      if (Array.isArray(byTitle) && byTitle.length) chunks = byTitle;
    }

    if (!chunks.length) {
      retrievalDebug.retrievalMode = "vector";
      const embed = await openai.embeddings.create({ model: "text-embedding-3-small", input: q });
      const query_embedding = embed.data[0].embedding;
      const { data: vs, error: matchErr } = await supabase.rpc("match_paper_chunks", {
        query_embedding,
        match_count: topK,
      });
      if (matchErr) return json(res, 500, { error: `Supabase RPC error: ${matchErr.message}` });
      chunks = vs || [];
    }

    if (needHeader && chunks.length) {
      const topSourceId = chunks[0]?.source_id;
      if (topSourceId) {
        const haveIdx = new Set((chunks || []).map((c) => String(c.source_id) + ":" + String(c.chunk_index)));
        const { data: headRows, error: headErr } = await supabase
          .from("paper_chunks")
          .select("source_id, source_title, source_year, source_url, chunk_index, content")
          .eq("source_id", topSourceId)
          .in("chunk_index", [0, 1])
          .order("chunk_index", { ascending: true });
        if (headErr) return json(res, 500, { error: `Supabase header fetch error: ${headErr.message}` });
        for (const r of headRows || []) {
          const k = String(r.source_id) + ":" + String(r.chunk_index);
          if (!haveIdx.has(k)) chunks.unshift(r);
        }
      }
    }

    const citations = (chunks || []).map((c, i) => ({
      ref: `ref:${i + 1}`,
      source_title: c.source_title,
      source_year: c.source_year,
      source_url: c.source_url,
      source_id: c.source_id,
      chunk_index: c.chunk_index,
      similarity: c.similarity,
      excerpt: c.content,
    }));

    const contextBlock =
      citations.length === 0
        ? "No paper excerpts were retrieved."
        : citations
            .map((c) => {
              const head = `[${c.ref}] ${c.source_title || c.source_id}${c.source_year ? ` (${c.source_year})` : ""}${
                c.source_url ? ` ${c.source_url}` : ""
              }`;
              return `${head}\n${String(c.excerpt || "").slice(0, 1400)}`;
            })
            .join("\n\n---\n\n");

    const ragRules = buildRagRules();
    const finalSystem =
      (systemPrompt ? systemPrompt + "\n\n" + ragRules : ragRules) +
      "\n\nPaper excerpts:\n" +
      contextBlock +
      "\n\nWhen excerpts are relevant, ground your answer in them and cite [ref:n].";

    const safeMsgs = Array.isArray(messages) ? messages.filter((m) => m?.role !== "system") : [];
    const chatMessages = [{ role: "system", content: finalSystem }, ...(safeMsgs.length ? safeMsgs : [{ role: "user", content: q }])];

    const completion = await openai.chat.completions.create({
      model,
      messages: chatMessages,
      temperature,
      max_tokens: maxTokens,
    });

    const content = completion.choices[0]?.message?.content ?? "";
    const includeDebug = debug || citations.length === 0;
    return json(res, 200, { content, citations, ...(includeDebug ? { retrievalDebug } : {}) });
  } catch (e) {
    return json(res, 500, { error: String(e?.message || e) });
  }
});

server.listen(PORT, "127.0.0.1", () => {
  // eslint-disable-next-line no-console
  console.log(`Local RAG server listening on http://localhost:${PORT}/api/chat`);
});

