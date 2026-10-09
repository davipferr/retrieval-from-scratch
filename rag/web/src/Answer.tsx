import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Turns `[2]` citations into links to `#cite-2`, outside fenced code blocks
 * (where `[2]` is far more likely to be an array index).
 */
function linkCitations(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?(?:```|$))/)
    .map((part, i) => (i % 2 === 1 ? part : part.replace(/\[(\d{1,2})\](?![(:])/g, "[$1](#cite-$1)")))
    .join("");
}

export function Answer({ text, onCite }: { text: string; onCite: (n: number) => void }) {
  return (
    <div className="markdown">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const cite = href?.match(/^#cite-(\d+)$/);
            if (cite) {
              const n = Number(cite[1]);
              return (
                <button type="button" className="cite" onClick={() => onCite(n)} title={`Show source ${n}`}>
                  {n}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {linkCitations(text)}
      </Markdown>
    </div>
  );
}
