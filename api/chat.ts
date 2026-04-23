export const config = { runtime: "nodejs" };

import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

type Msg = { role: "system" | "user" | "assistant"; content: string };

function extractQuotedTitle(q: string): string | null {
  const s = String(q || "");
  // Match “Title”, "Title", ‘Title’, 'Title'
  const m = s.match(/[“"'‘]([^”"'’]{18,260})[”"'’]/);
  if (!m) return null;
  const title = m[1].trim();
  // Avoid grabbing short generic phrases.
  if (title.split(/\s+/).length < 4) return null;
  return title;
}

function setCors(res: any) {
  res.setHeader("access-control-allow-origin", "*");
  res.setHeader("access-control-allow-methods", "POST, OPTIONS");
  res.setHeader("access-control-allow-headers", "content-type");
}

module.exports = async function handler(req: any, res: any) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(204).json({});
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
    const SUPABASE_URL = process.env.SUPABASE_URL;
    const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;
    const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!OPENAI_API_KEY) return res.status(500).json({ error: "Missing OPENAI_API_KEY" });
    if (!SUPABASE_URL) return res.status(500).json({ error: "Missing SUPABASE_URL" });
    if (!SUPABASE_ANON_KEY && !SUPABASE_SERVICE_ROLE_KEY) {
      return res.status(500).json({ error: "Missing SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)" });
    }

    const body = req.body || {};
    const question = String(body.question || "").trim();
    const messages = (body.messages || []) as Msg[];
    const systemPrompt = String(body.systemPrompt || "").trim();
    const model = String(body.model || "gpt-4o-mini");
    const topK = Number(body.topK || 8);

    if (!question && (!Array.isArray(messages) || messages.length === 0)) {
      return res.status(400).json({ error: "Provide question or messages" });
    }

    const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY || SUPABASE_ANON_KEY!);

    const q = question || messages[messages.length - 1]?.content || "";
    const quotedTitle = extractQuotedTitle(q);

    let chunks: any[] = [];
    // If user names a specific paper title, do a metadata-directed fetch first.
    if (quotedTitle) {
      const { data: byTitle, error: titleErr } = await supabase
        .from("paper_chunks")
        .select("source_id, source_title, source_year, source_url, chunk_index, content")
        .ilike("source_title", `%${quotedTitle}%`)
        .order("chunk_index", { ascending: true })
        .limit(Math.max(6, Math.min(30, topK)));

      if (titleErr) return res.status(500).json({ error: `Supabase title lookup error: ${titleErr.message}` });
      if (Array.isArray(byTitle) && byTitle.length) {
        chunks = byTitle;
      }
    }

    // Fall back to vector search if no title match.
    if (!chunks.length) {
      const embed = await openai.embeddings.create({ model: "text-embedding-3-small", input: q });
      const query_embedding = embed.data[0].embedding;

      const { data: vs, error: matchErr } = await supabase.rpc("match_paper_chunks", {
        query_embedding,
        match_count: topK,
      });
      if (matchErr) return res.status(500).json({ error: `Supabase RPC error: ${matchErr.message}` });
      chunks = vs || [];
    }

    const citations = (chunks || []).map((c: any, i: number) => ({
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
            .map((c: any) => {
              const head = `[${c.ref}] ${c.source_title || c.source_id}${c.source_year ? ` (${c.source_year})` : ""}${
                c.source_url ? ` ${c.source_url}` : ""
              }`;
              return `${head}\n${String(c.excerpt || "").slice(0, 1400)}`;
            })
            .join("\n\n---\n\n");

    const finalSystem =
      (systemPrompt ||
        "You are a helpful assistant. Use the provided paper excerpts when relevant and cite with [ref:1], [ref:2]. Answer general questions directly; do not refuse with a narrow 'scope' message.") +
      "\n\nPaper excerpts:\n" +
      contextBlock +
      "\n\nCite sources like [ref:1], [ref:2] after claims when the excerpts support a statement. If the question is off-topic for the excerpts, answer from general knowledge and say the excerpts are not on point.";

    const chatMessages: Msg[] = [{ role: "system", content: finalSystem }];
    if (Array.isArray(messages) && messages.length) chatMessages.push(...messages);
    else chatMessages.push({ role: "user", content: q });

    const completion = await openai.chat.completions.create({
      model,
      messages: chatMessages,
      temperature: Number(body.temperature ?? 0.3),
      max_tokens: Number(body.maxTokens ?? 900),
    });

    const content = completion.choices[0]?.message?.content ?? "";
    return res.status(200).json({ content, citations });
  } catch (e: any) {
    return res.status(500).json({ error: String(e?.message || e) });
  }
};

