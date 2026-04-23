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

function extractParenTitle(q: string): string | null {
  const s = String(q || "");
  // Match: <title> (2024)
  const m = s.match(/([A-Za-z0-9][\s\S]{18,260}?)\s*\(\s*(19\d{2}|20\d{2})\s*\)\s*$/);
  if (!m) return null;
  const title = m[1].trim();
  if (title.split(/\s+/).length < 4) return null;
  return title;
}

function wantsAuthorOrAbstract(q: string): boolean {
  const s = String(q || "").toLowerCase();
  return /\b(author|authors|who wrote|written by|abstract)\b/.test(s);
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
    const model = String(body.model || "gpt-4o");
    const topK = Number(body.topK || 8);

    if (!question && (!Array.isArray(messages) || messages.length === 0)) {
      return res.status(400).json({ error: "Provide question or messages" });
    }

    const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY || SUPABASE_ANON_KEY!);

    const q = question || messages[messages.length - 1]?.content || "";
    const quotedTitle = extractQuotedTitle(q);
    const parenTitle = extractParenTitle(q);
    const titleHint = quotedTitle || parenTitle;
    const needHeader = wantsAuthorOrAbstract(q);

    let chunks: any[] = [];
    // If user names a specific paper title, do a metadata-directed fetch first.
    if (titleHint) {
      const { data: byTitle, error: titleErr } = await supabase
        .from("paper_chunks")
        .select("source_id, source_title, source_year, source_url, chunk_index, content")
        .ilike("source_title", `%${titleHint}%`)
        .order("chunk_index", { ascending: true })
        .limit(Math.max(12, Math.min(40, topK * 2)));

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

    // If the user asks for authors/abstract, force-include the header chunk(s) for the top retrieved paper.
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
        if (headErr) return res.status(500).json({ error: `Supabase header fetch error: ${headErr.message}` });
        for (const r of headRows || []) {
          const k = String(r.source_id) + ":" + String(r.chunk_index);
          if (!haveIdx.has(k)) chunks.unshift(r);
        }
      }
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
        [
          "You are a helpful assistant with access to the section 'Paper excerpts' below.",
          "",
          "Hard rules:",
          "- Do NOT say you 'don't have access' or 'can't access the paper' when excerpts are present. You DO have access to those excerpts.",
          "- Do NOT deflect with scope refusals (e.g. 'I can only help with urban service equity'). Answer the user's question directly.",
          "",
          "How to answer using excerpts:",
          "- If the user asks about a specific paper (title/authors/year/venue/claims) and the relevant info appears in the excerpts, QUOTE the exact line(s) and cite them, e.g. [ref:1].",
          "- If the user asks for the full paper text, do NOT reproduce the full paper. Instead: say you can't provide full text, then summarize what the excerpts say and include 1–3 short quoted snippets with citations.",
          "- If excerpts do not contain the requested detail (e.g., author line missing), say that plainly and suggest what you *can* do (e.g., ask to re-index first page / provide DOI).",
          "",
          "Citations:",
          "- Put citations right after the sentence they support, like [ref:2].",
          "- Never invent citations. If a claim is general knowledge, do not cite.",
        ].join("\n")) +
      "\n\nPaper excerpts:\n" +
      contextBlock +
      "\n\nWhen excerpts are relevant, ground your answer in them and cite [ref:n]. If excerpts are off-topic, answer from general knowledge in one concise response and (optionally) mention the excerpts are not on point.";

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

