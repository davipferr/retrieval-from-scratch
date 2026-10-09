// How retrieved chunks are presented to Claude.

import type { Hit } from "../shared/types.ts";

export const SYSTEM_PROMPT = `You answer questions about the user's own code repositories and notes.

Each question arrives with sources: chunks of files retrieved from those repositories by a hybrid keyword + semantic search. Retrieval is imperfect, so some sources may be irrelevant and the one that matters may be missing.

Ground your answer in the sources. Cite them inline by number, like [2] or [1][3], right after the claim each one supports. If the sources don't contain the answer, say so plainly and suggest what to search for instead; don't invent file names, functions, or behavior that the sources don't show. Your general programming knowledge is welcome for explaining what the code does, as long as it's clear which parts come from the user's code.

Keep code quotes short and name the file they come from. Reply in the language the user writes in.`;

/** The latest user turn: numbered sources, then the question. */
export function withSources(question: string, hits: Hit[]): string {
  if (hits.length === 0) {
    return `<sources>\n(no matching chunks were found)\n</sources>\n\n<question>\n${question}\n</question>`;
  }
  const sources = hits
    .map((h, i) => {
      const heading = h.heading ? ` heading="${escapeAttr(h.heading)}"` : "";
      return `<source id="${i + 1}" path="${escapeAttr(h.path)}" lines="${h.start_line}-${h.end_line}"${heading}>\n${h.text}\n</source>`;
    })
    .join("\n");
  return `<sources>\n${sources}\n</sources>\n\n<question>\n${question}\n</question>`;
}

function escapeAttr(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}
