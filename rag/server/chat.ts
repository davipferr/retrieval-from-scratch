// The RAG loop: retrieve -> build the prompt -> stream Claude's answer.

import Anthropic from "@anthropic-ai/sdk";

import type { ChatEvent, ChatRequest } from "../shared/types.ts";
import { SYSTEM_PROMPT, withSources } from "./prompt.ts";
import { retrievalQuery, search } from "./retrieval.ts";

let client: Anthropic | undefined;

function getClient(): Anthropic {
  // Created on first use, so the server starts (and retrieval works) even
  // before credentials are configured. Resolves ANTHROPIC_API_KEY,
  // ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
  client ??= new Anthropic();
  return client;
}

export async function* chat(req: ChatRequest, signal: AbortSignal): AsyncGenerator<ChatEvent> {
  // 1. Retrieve.
  const query = retrievalQuery(req.messages);
  const found = await search(query, req.mode, req.k, signal);
  yield { type: "sources", query, hits: found.hits, timings: found.timings };

  // 2. Prompt: earlier turns as plain text; sources go only on the latest
  //    question, so old turns' chunks don't pile up in the context.
  const last = req.messages.length - 1;
  const messages: Anthropic.Beta.BetaMessageParam[] = req.messages.map((m, i) => ({
    role: m.role,
    content: i === last ? withSources(m.content, found.hits) : m.content,
  }));

  // 3. Stream the answer.
  const stream = getClient().beta.messages.stream(
    {
      model: req.model,
      max_tokens: 64000,
      system: SYSTEM_PROMPT,
      messages,
      // Thinking is always on for these models; "summarized" streams a readable
      // summary of it instead of a silent pause before the answer.
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: req.effort },
      // If a safety classifier declines the request, the API retries it on
      // the fallback model Anthropic recommends for that refusal category,
      // within the same stream.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
    },
    { signal },
  );

  for await (const event of stream) {
    if (event.type === "content_block_start" && event.content_block.type === "fallback") {
      const block = event.content_block;
      yield { type: "fallback", from: block.from.model, to: block.to.model };
    } else if (event.type === "content_block_delta") {
      if (event.delta.type === "text_delta") yield { type: "text", text: event.delta.text };
      else if (event.delta.type === "thinking_delta") yield { type: "thinking", text: event.delta.thinking };
    }
  }

  const final = await stream.finalMessage();
  yield {
    type: "done",
    model: final.model,
    // "refusal" here means the whole fallback chain declined.
    stop_reason: final.stop_reason,
    refusal_category: final.stop_reason === "refusal" ? (final.stop_details?.category ?? null) : null,
    usage: { input_tokens: final.usage.input_tokens, output_tokens: final.usage.output_tokens },
  };
}
