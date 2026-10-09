//! The inverted index: term -> list of documents containing it.
//!
//! Building happens in memory (a `HashMap<term, Vec<Posting>>`), then
//! everything is written to a directory as flat files that are memory-mapped
//! at query time. Nothing is deserialized up front except the small document
//! table: looking up a term is a binary search directly over the mapped bytes,
//! and the OS pages in only the parts we touch.
//!
//! ```text
//! index/
//!   terms.idx     header + one fixed-size (24 byte) entry per term, sorted by term
//!   terms.dat     all term strings concatenated (entries point into it)
//!   postings.dat  all posting lists, delta + varint compressed (see `postings`)
//!   docs.dat      stored document text, for result snippets
//!   docs.json     per-document metadata: url, title, length, where its text is
//! ```
//!
//! `terms.idx` layout (all little-endian):
//!
//! ```text
//! header: magic "MSETERM1" (8 bytes) | term count u64
//! entry:  term_offset u32 | term_len u32 | postings_offset u64 | postings_len u32 | doc_freq u32
//! ```
//!
//! Fixed-size entries are what make binary search over a memory map possible:
//! entry `i` lives at byte `16 + 24 * i`.

use std::collections::HashMap;
use std::fs::{self, File};
use std::ops::Deref;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::levenshtein::SortedTerms;
use crate::postings::{self, Posting, PostingIter};
use crate::tokenizer;

const MAGIC: &[u8; 8] = b"MSETERM1";
const HEADER_LEN: usize = 16;
const ENTRY_LEN: usize = 24;

/// Title words count this many times, so a match in the title outranks one in the body.
const TITLE_BOOST: u32 = 3;
/// Cap on stored text per document (only used for snippets).
const MAX_STORED_TEXT: usize = 64 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub url: String,
    pub title: String,
    /// Number of indexed terms (with the title boost), for BM25 length normalization.
    pub len: u32,
    text_offset: u64,
    text_len: u32,
}

#[derive(Serialize, Deserialize)]
struct DocTable {
    avg_len: f32,
    docs: Vec<DocMeta>,
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct IndexBuilder {
    terms: HashMap<String, Vec<Posting>>,
    docs: Vec<DocMeta>,
    text: Vec<u8>,
    total_len: u64,
}

/// What `write` produced, to see the effect of compression.
#[derive(Debug)]
pub struct WriteStats {
    pub docs: usize,
    pub terms: usize,
    pub postings: usize,
    /// Size of the posting lists as two plain u32 per posting.
    pub postings_raw_bytes: usize,
    pub postings_bytes: usize,
}

impl IndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Analyzes and adds a document. Doc ids are assigned in insertion order,
    /// which keeps every posting list sorted without any extra work.
    pub fn add_document(&mut self, url: &str, title: &str, body: &str) -> u32 {
        let doc = self.docs.len() as u32;

        let mut counts: HashMap<String, u32> = HashMap::new();
        let mut len = 0u32;
        for term in tokenizer::analyze(title) {
            *counts.entry(term).or_default() += TITLE_BOOST;
            len += TITLE_BOOST;
        }
        for term in tokenizer::analyze(body) {
            *counts.entry(term).or_default() += 1;
            len += 1;
        }
        for (term, tf) in counts {
            self.terms
                .entry(term)
                .or_default()
                .push(Posting { doc, tf });
        }

        let mut stored_end = body.len().min(MAX_STORED_TEXT);
        while !body.is_char_boundary(stored_end) {
            stored_end -= 1;
        }
        let text_offset = self.text.len() as u64;
        self.text.extend_from_slice(&body.as_bytes()[..stored_end]);

        self.total_len += u64::from(len);
        self.docs.push(DocMeta {
            url: url.to_string(),
            title: title.to_string(),
            len,
            text_offset,
            text_len: stored_end as u32,
        });
        doc
    }

    pub fn write(self, dir: &Path) -> Result<WriteStats> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

        let mut terms: Vec<(String, Vec<Posting>)> = self.terms.into_iter().collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut table = Vec::with_capacity(HEADER_LEN + ENTRY_LEN * terms.len());
        table.extend_from_slice(MAGIC);
        table.extend_from_slice(&(terms.len() as u64).to_le_bytes());
        let mut term_bytes = Vec::new();
        let mut postings_buf = Vec::new();
        let mut n_postings = 0;

        for (term, list) in &terms {
            let term_offset = term_bytes.len() as u32;
            term_bytes.extend_from_slice(term.as_bytes());
            let postings_offset = postings_buf.len() as u64;
            postings::encode(list, &mut postings_buf);
            let postings_len = (postings_buf.len() as u64 - postings_offset) as u32;
            n_postings += list.len();

            table.extend_from_slice(&term_offset.to_le_bytes());
            table.extend_from_slice(&(term.len() as u32).to_le_bytes());
            table.extend_from_slice(&postings_offset.to_le_bytes());
            table.extend_from_slice(&postings_len.to_le_bytes());
            table.extend_from_slice(&(list.len() as u32).to_le_bytes());
        }

        let avg_len = if self.docs.is_empty() {
            0.0
        } else {
            self.total_len as f32 / self.docs.len() as f32
        };
        let doc_table = DocTable {
            avg_len,
            docs: self.docs,
        };

        fs::write(dir.join("terms.idx"), &table)?;
        fs::write(dir.join("terms.dat"), &term_bytes)?;
        fs::write(dir.join("postings.dat"), &postings_buf)?;
        fs::write(dir.join("docs.dat"), &self.text)?;
        fs::write(dir.join("docs.json"), serde_json::to_vec(&doc_table)?)?;

        Ok(WriteStats {
            docs: doc_table.docs.len(),
            terms: terms.len(),
            postings: n_postings,
            postings_raw_bytes: n_postings * 8,
            postings_bytes: postings_buf.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// A memory-mapped file. Zero-length files cannot be mapped (Windows refuses),
/// so they are represented separately.
enum Bytes {
    Mapped(Mmap),
    Empty,
}

impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Mapped(m) => m,
            Bytes::Empty => &[],
        }
    }
}

fn map_file(path: &Path) -> Result<Bytes> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    if file.metadata()?.len() == 0 {
        return Ok(Bytes::Empty);
    }
    // SAFETY: the index files are written once and never modified while
    // mapped. If another process truncated them underneath us, reads could
    // fault — the usual caveat of memory-mapped files.
    let map = unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
    Ok(Bytes::Mapped(map))
}

#[derive(Debug, Clone, Copy)]
struct TermEntry {
    term_offset: usize,
    term_len: usize,
    postings_offset: usize,
    postings_len: usize,
    doc_freq: u32,
}

pub struct Index {
    table: Bytes,
    term_bytes: Bytes,
    postings: Bytes,
    text: Bytes,
    docs: Vec<DocMeta>,
    avg_len: f32,
}

fn u32_at(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().unwrap())
}

fn u64_at(buf: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(buf[at..at + 8].try_into().unwrap())
}

impl Index {
    pub fn open(dir: &Path) -> Result<Index> {
        if !dir.join("terms.idx").exists() {
            bail!(
                "no index found in {} (build one with `crawl` or `index`)",
                dir.display()
            );
        }
        let table = map_file(&dir.join("terms.idx"))?;
        ensure!(
            table.len() >= HEADER_LEN && &table[..8] == MAGIC,
            "terms.idx: bad header"
        );
        let n_terms = u64_at(&table, 8) as usize;
        ensure!(
            table.len() == HEADER_LEN + n_terms * ENTRY_LEN,
            "terms.idx: wrong length"
        );

        let doc_table: DocTable = serde_json::from_slice(&fs::read(dir.join("docs.json"))?)
            .context("parsing docs.json")?;

        let index = Index {
            table,
            term_bytes: map_file(&dir.join("terms.dat"))?,
            postings: map_file(&dir.join("postings.dat"))?,
            text: map_file(&dir.join("docs.dat"))?,
            docs: doc_table.docs,
            avg_len: doc_table.avg_len,
        };

        // Validate every entry once, so the accessors below can't go out of bounds.
        for i in 0..n_terms {
            let e = index.entry(i);
            ensure!(
                e.term_offset + e.term_len <= index.term_bytes.len(),
                "term {i} out of range"
            );
            ensure!(
                e.postings_offset + e.postings_len <= index.postings.len(),
                "postings for term {i} out of range"
            );
            std::str::from_utf8(&index.term_bytes[e.term_offset..e.term_offset + e.term_len])
                .with_context(|| format!("term {i} is not UTF-8"))?;
        }
        for (i, d) in index.docs.iter().enumerate() {
            ensure!(
                d.text_offset as usize + d.text_len as usize <= index.text.len(),
                "text of doc {i} out of range"
            );
        }
        Ok(index)
    }

    pub fn num_docs(&self) -> usize {
        self.docs.len()
    }

    pub fn num_terms(&self) -> usize {
        (self.table.len() - HEADER_LEN) / ENTRY_LEN
    }

    pub fn avg_doc_len(&self) -> f32 {
        self.avg_len
    }

    pub fn doc(&self, id: u32) -> &DocMeta {
        &self.docs[id as usize]
    }

    pub fn doc_text(&self, id: u32) -> &str {
        let d = &self.docs[id as usize];
        let start = d.text_offset as usize;
        std::str::from_utf8(&self.text[start..start + d.text_len as usize]).unwrap_or("")
    }

    fn entry(&self, i: usize) -> TermEntry {
        let at = HEADER_LEN + i * ENTRY_LEN;
        let t = &self.table;
        TermEntry {
            term_offset: u32_at(t, at) as usize,
            term_len: u32_at(t, at + 4) as usize,
            postings_offset: u64_at(t, at + 8) as usize,
            postings_len: u32_at(t, at + 16) as usize,
            doc_freq: u32_at(t, at + 20),
        }
    }

    pub fn term(&self, i: usize) -> &str {
        let e = self.entry(i);
        let bytes = &self.term_bytes[e.term_offset..e.term_offset + e.term_len];
        // SAFETY: `open` checked that every term is valid UTF-8. This is the
        // hottest function in fuzzy search, so we don't re-validate each call.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Binary search over the sorted, memory-mapped term table.
    pub fn find(&self, term: &str) -> Option<usize> {
        let (mut lo, mut hi) = (0, self.num_terms());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.term(mid).cmp(term) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Number of documents containing term `i`.
    pub fn doc_freq(&self, i: usize) -> u32 {
        self.entry(i).doc_freq
    }

    pub fn postings(&self, i: usize) -> PostingIter<'_> {
        let e = self.entry(i);
        let data = &self.postings[e.postings_offset..e.postings_offset + e.postings_len];
        PostingIter::new(data, e.doc_freq)
    }

    /// Size in bytes of each file, for `stats`.
    pub fn sizes(&self) -> [(&'static str, usize); 4] {
        [
            ("terms.idx", self.table.len()),
            ("terms.dat", self.term_bytes.len()),
            ("postings.dat", self.postings.len()),
            ("docs.dat", self.text.len()),
        ]
    }
}

impl SortedTerms for Index {
    fn len(&self) -> usize {
        self.num_terms()
    }
    fn term(&self, i: usize) -> &str {
        Index::term(self, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mini_search_test_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn write_then_read() {
        let mut b = IndexBuilder::new();
        b.add_document("a", "Rust", "Rust is a systems programming language.");
        b.add_document(
            "b",
            "Cooking",
            "A recipe for pão de queijo. No rust here... well, one rust.",
        );
        b.add_document("c", "Empty", "");
        let dir = temp_dir("write_then_read");
        let stats = b.write(&dir).unwrap();
        assert_eq!(stats.docs, 3);

        let index = Index::open(&dir).unwrap();
        assert_eq!(index.num_docs(), 3);
        let rust = index.find("rust").expect("rust is indexed");
        assert_eq!(index.doc_freq(rust), 2);
        let postings: Vec<Posting> = index.postings(rust).collect();
        // doc 0: 3 (title boost) + 1 (body); doc 1: 2 in body
        assert_eq!(
            postings,
            [Posting { doc: 0, tf: 4 }, Posting { doc: 1, tf: 2 }]
        );
        assert!(index.find("pao").is_some(), "accents are folded");
        assert!(index.find("the").is_none(), "stopwords are dropped");
        assert_eq!(
            index.doc_text(1),
            "A recipe for pão de queijo. No rust here... well, one rust."
        );

        let terms: Vec<&str> = (0..index.num_terms()).map(|i| index.term(i)).collect();
        assert!(terms.windows(2).all(|w| w[0] < w[1]), "terms are sorted");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn empty_index() {
        let dir = temp_dir("empty_index");
        IndexBuilder::new().write(&dir).unwrap();
        let index = Index::open(&dir).unwrap();
        assert_eq!(index.num_terms(), 0);
        assert_eq!(index.find("anything"), None);
        fs::remove_dir_all(dir).ok();
    }
}
