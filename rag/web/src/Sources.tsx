import { useEffect, useRef } from "react";

import type { Hit, Mode, Timings } from "../../shared/types.ts";

export interface Retrieved {
  query: string;
  hits: Hit[];
  timings: Timings;
}

function Rank({ label, rank }: { label: string; rank: number | null }) {
  return (
    <span className={`rank ${rank ? "" : "rank-miss"}`} title={rank ? `#${rank} in ${label} results` : `not in ${label} top 50`}>
      {label} {rank ? `#${rank}` : "–"}
    </span>
  );
}

export function Sources({
  retrieved,
  mode,
  highlight,
}: {
  retrieved: Retrieved | undefined;
  mode: Mode;
  highlight: number | null;
}) {
  const refs = useRef<(HTMLElement | null)[]>([]);
  useEffect(() => {
    if (highlight) refs.current[highlight - 1]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [highlight, retrieved]);

  if (!retrieved) {
    return (
      <aside className="sources">
        <h2>Sources</h2>
        <p className="muted">Retrieved chunks for the selected answer appear here.</p>
      </aside>
    );
  }
  const t = retrieved.timings;
  return (
    <aside className="sources">
      <h2>Sources</h2>
      <p className="muted small">
        “{retrieved.query}” · {mode} · {t.total_ms.toFixed(1)} ms
        {t.embed_ms > 0 && ` (embed ${t.embed_ms.toFixed(1)}, hnsw ${t.vector_ms.toFixed(1)}, bm25 ${t.bm25_ms.toFixed(1)})`}
      </p>
      {retrieved.hits.length === 0 && <p className="muted">No chunks matched.</p>}
      <ol className="source-list">
        {retrieved.hits.map((h, i) => (
          <li
            key={h.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            className={`source ${highlight === i + 1 ? "source-active" : ""}`}
          >
            <div className="source-head">
              <span className="source-num">{i + 1}</span>
              <span className="source-path" title={h.abs_path}>
                {h.path}
                <span className="muted">:{h.start_line}-{h.end_line}</span>
              </span>
            </div>
            {h.heading && <div className="source-heading">§ {h.heading}</div>}
            <div className="ranks">
              <Rank label="BM25" rank={h.bm25_rank} />
              <Rank label="vector" rank={h.vector_rank} />
              {h.similarity !== null && <span className="rank">cos {h.similarity.toFixed(2)}</span>}
            </div>
            <pre className="source-text">{h.text}</pre>
          </li>
        ))}
      </ol>
    </aside>
  );
}
