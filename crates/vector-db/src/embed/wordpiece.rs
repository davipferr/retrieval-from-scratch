//! BERT's WordPiece tokenizer (uncased), as used by bge-small-en and MiniLM.
//!
//! The model only understands ids from its 30,522-entry vocabulary, so text
//! goes through two stages:
//!
//! 1. **Basic tokenization**: lowercase, strip accents (reusing the search
//!    engine's `normalize`), split on whitespace, and make every punctuation
//!    character its own token: `"Don't panic!"` → `don ' t panic !`.
//! 2. **WordPiece**: split each word into the longest vocabulary pieces,
//!    greedily from the left. Pieces after the first are prefixed with `##`:
//!    `"tokenization"` → `token ##ization`; `"hnsw"` → `h ##ns ##w`.
//!    Any word can be spelled out of pieces, so there is (almost) never an
//!    unknown word, just more pieces for rarer ones.
//!
//! Finally the sequence is wrapped as `[CLS] … [SEP]`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use search_engine::tokenizer::normalize;

const MAX_WORD_CHARS: usize = 100;

pub struct WordPiece {
    vocab: HashMap<String, u32>,
    cls: u32,
    sep: u32,
    unk: u32,
}

impl WordPiece {
    /// Loads a `vocab.txt`: one token per line, the line number is the id.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_tokens(text.lines())
    }

    pub fn from_tokens<'a>(tokens: impl IntoIterator<Item = &'a str>) -> Result<Self> {
        let vocab: HashMap<String, u32> = tokens
            .into_iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        let id = |t: &str| {
            vocab
                .get(t)
                .copied()
                .with_context(|| format!("vocab has no {t}"))
        };
        Ok(Self {
            cls: id("[CLS]")?,
            sep: id("[SEP]")?,
            unk: id("[UNK]")?,
            vocab,
        })
    }

    /// Token ids for `text`, including `[CLS]`/`[SEP]`, at most `max_len` long.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let mut ids = vec![self.cls];
        for word in basic_tokens(text) {
            if ids.len() >= max_len - 1 {
                break;
            }
            self.wordpiece(&word, &mut ids);
        }
        ids.truncate(max_len - 1);
        ids.push(self.sep);
        ids
    }

    /// Readable pieces, for debugging (`vecdb tokenize`).
    pub fn pieces(&self, text: &str) -> Vec<String> {
        let by_id: HashMap<u32, &str> = self.vocab.iter().map(|(t, &i)| (i, t.as_str())).collect();
        self.encode(text, usize::MAX)
            .iter()
            .map(|i| by_id[i].to_string())
            .collect()
    }

    /// Greedy longest-match-first.
    fn wordpiece(&self, word: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > MAX_WORD_CHARS {
            out.push(self.unk);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut candidate = String::new();
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                candidate.clear();
                if start > 0 {
                    candidate.push_str("##");
                }
                candidate.extend(&chars[start..end]);
                if let Some(&id) = self.vocab.get(&candidate) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            let Some(id) = found else {
                // no piece matches at this position: the whole word is unknown
                out.push(self.unk);
                return;
            };
            pieces.push(id);
            start = end;
        }
        out.extend(pieces);
    }
}

/// Lowercased, accent-stripped words, with punctuation split off.
fn basic_tokens(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for c in normalize(text).chars() {
        if c.is_whitespace() || c.is_control() || c == '\u{fffd}' {
            flush(&mut current, &mut words);
        } else if is_punctuation(c) || is_cjk(c) {
            // punctuation and CJK ideographs are tokens on their own
            flush(&mut current, &mut words);
            words.push(c.to_string());
        } else {
            current.push(c);
        }
    }
    flush(&mut current, &mut words);
    words
}

fn flush(current: &mut String, words: &mut Vec<String>) {
    if !current.is_empty() {
        words.push(std::mem::take(current));
    }
}

/// BERT treats all non-alphanumeric ASCII as punctuation (so `$`, `^`, `` ` ``
/// split too), plus Unicode punctuation; this covers the common blocks.
fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(c, '\u{2000}'..='\u{206f}' | '\u{3000}'..='\u{303f}' | '¡' | '¿' | '«' | '»' | '·')
}

fn is_cjk(c: char) -> bool {
    matches!(c, '\u{4e00}'..='\u{9fff}' | '\u{3400}'..='\u{4dbf}' | '\u{f900}'..='\u{faff}')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> WordPiece {
        let vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "token",
            "##ization",
            "##s",
            "don",
            "'",
            "t",
            "panic",
            "!",
            "h",
            "##ns",
            "##w",
            "cafe",
        ];
        WordPiece::from_tokens(vocab).unwrap()
    }

    #[test]
    fn splits_punctuation_and_pieces() {
        let wp = tiny();
        assert_eq!(
            wp.pieces("Don't panic! Tokenizations"),
            [
                "[CLS]",
                "don",
                "'",
                "t",
                "panic",
                "!",
                "token",
                "##ization",
                "##s",
                "[SEP]"
            ]
        );
        assert_eq!(
            wp.pieces("HNSW café"),
            ["[CLS]", "h", "##ns", "##w", "cafe", "[SEP]"]
        );
    }

    #[test]
    fn unknown_words_and_truncation() {
        let wp = tiny();
        assert_eq!(wp.pieces("xyz"), ["[CLS]", "[UNK]", "[SEP]"]);
        let ids = wp.encode("panic panic panic panic panic", 4);
        assert_eq!(ids.len(), 4);
        assert_eq!(ids, [2, 10, 10, 3]);
    }
}
