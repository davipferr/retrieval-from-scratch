//! Splitting files into chunks for retrieval.
//!
//! Why chunk at all? An embedding squeezes a whole text into one vector; a
//! 2,000-line file averages too many topics to match any question well, and
//! the model only reads its first 512 tokens anyway. And for RAG, retrieved
//! text gets pasted into the prompt, where every irrelevant line costs tokens
//! and distracts the model.
//!
//! Strategy, line-based so every chunk maps back to `path:start-end`:
//!
//! 1. Split the file into **blocks** at blank lines. In Markdown a heading
//!    always starts a new block. Blank lines are where code and prose
//!    naturally separate (between functions, between paragraphs).
//! 2. **Pack** consecutive blocks into a chunk until adding the next would pass
//!    `TARGET_CHARS`, so small blocks share a chunk instead of becoming
//!    two-line fragments.
//! 3. A block bigger than `MAX_CHARS` on its own (a huge function) is cut
//!    by lines.
//!
//! Each chunk remembers the Markdown heading it falls under, which gets
//! prepended when embedding: "## Installation" gives the paragraph below it
//! meaning it doesn't carry by itself.

use serde::{Deserialize, Serialize};

/// ~250 tokens: comfortably inside the model's 512-token window.
pub const TARGET_CHARS: usize = 1000;
pub const MAX_CHARS: usize = 1600;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// 1-based, inclusive.
    pub start_line: usize,
    pub end_line: usize,
    /// Nearest Markdown heading above the chunk, if any.
    pub heading: Option<String>,
    pub text: String,
}

struct Block<'a> {
    start: usize, // 0-based line index
    lines: Vec<&'a str>,
    heading: Option<String>,
}

impl Block<'_> {
    fn chars(&self) -> usize {
        self.lines.iter().map(|l| l.len() + 1).sum()
    }
}

pub fn chunk_text(text: &str, markdown: bool) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current: Option<Block> = None;
    for block in blocks(text, markdown) {
        for block in split_oversized(block) {
            let fits = current.as_ref().is_some_and(|c| {
                c.chars() + block.chars() <= TARGET_CHARS
                    // don't merge across a heading change: different section
                    && c.heading == block.heading
            });
            if fits {
                let c = current.as_mut().unwrap();
                // keep the blank lines between blocks so line numbers stay true
                let gap = block.start - (c.start + c.lines.len());
                c.lines.extend(std::iter::repeat_n("", gap));
                c.lines.extend(block.lines);
            } else {
                if let Some(done) = current.take() {
                    chunks.push(finish(done));
                }
                current = Some(block);
            }
        }
    }
    if let Some(done) = current {
        chunks.push(finish(done));
    }
    chunks
}

fn finish(block: Block) -> Chunk {
    Chunk {
        start_line: block.start + 1,
        end_line: block.start + block.lines.len(),
        heading: block.heading,
        text: block.lines.join("\n"),
    }
}

/// Runs of non-blank lines.
fn blocks(text: &str, markdown: bool) -> Vec<Block<'_>> {
    let mut out: Vec<Block> = Vec::new();
    let mut heading: Option<String> = None;
    let mut in_fence = false;
    let mut current: Option<Block> = None;

    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        if markdown && trimmed.starts_with("```") {
            in_fence = !in_fence;
        }
        // `#` inside a code fence is a comment, not a heading
        let is_heading =
            markdown && !in_fence && trimmed.starts_with('#') && !trimmed.starts_with("#!");
        if (line.trim().is_empty() || is_heading)
            && let Some(b) = current.take()
        {
            out.push(b);
        }
        if is_heading {
            heading = Some(trimmed.trim_start_matches('#').trim().to_string());
        }
        if line.trim().is_empty() {
            continue;
        }
        current
            .get_or_insert_with(|| Block {
                start: i,
                lines: Vec::new(),
                heading: heading.clone(),
            })
            .lines
            .push(line);
    }
    out.extend(current);
    out
}

/// Cuts a block longer than `MAX_CHARS` into line ranges that fit
/// (a single enormous line, like minified code, is cut as is).
fn split_oversized(block: Block<'_>) -> Vec<Block<'_>> {
    if block.chars() <= MAX_CHARS {
        return vec![block];
    }
    let mut out = Vec::new();
    let mut piece = Block {
        start: block.start,
        lines: Vec::new(),
        heading: block.heading.clone(),
    };
    for (offset, line) in block.lines.iter().enumerate() {
        if !piece.lines.is_empty() && piece.chars() + line.len() + 1 > TARGET_CHARS {
            let next = Block {
                start: block.start + offset,
                lines: Vec::new(),
                heading: block.heading.clone(),
            };
            out.push(std::mem::replace(&mut piece, next));
        }
        piece.lines.push(line);
    }
    out.push(piece);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_blocks_are_packed_and_lines_are_exact() {
        let text = "fn a() {}\n\nfn b() {}\n\n\nfn c() {}\n";
        let chunks = chunk_text(text, false);
        assert_eq!(chunks.len(), 1);
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 6));
        assert_eq!(chunks[0].text, "fn a() {}\n\nfn b() {}\n\n\nfn c() {}");
    }

    #[test]
    fn markdown_headings_split_and_label() {
        let text = "# Title\nintro\n\n## Install\nrun cargo\n```sh\n# not a heading\n```\n\n## Usage\nsearch\n";
        let chunks = chunk_text(text, true);
        let summary: Vec<(Option<&str>, usize, usize)> = chunks
            .iter()
            .map(|c| (c.heading.as_deref(), c.start_line, c.end_line))
            .collect();
        assert_eq!(
            summary,
            [
                (Some("Title"), 1, 2),
                (Some("Install"), 4, 8),
                (Some("Usage"), 10, 11)
            ]
        );
    }

    #[test]
    fn big_blocks_are_cut() {
        let line = "x".repeat(99);
        let text = vec![line.as_str(); 50].join("\n"); // one 5,000-char block
        let chunks = chunk_text(&text, false);
        assert!(chunks.len() >= 5);
        assert!(chunks.iter().all(|c| c.text.len() <= MAX_CHARS));
        assert_eq!(chunks.first().unwrap().start_line, 1);
        assert_eq!(chunks.last().unwrap().end_line, 50);
        // contiguous, no gaps or overlaps
        assert!(
            chunks
                .windows(2)
                .all(|w| w[1].start_line == w[0].end_line + 1)
        );
    }
}
