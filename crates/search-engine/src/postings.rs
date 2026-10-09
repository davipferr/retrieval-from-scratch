//! Compressed posting lists.
//!
//! A posting list is the list of documents that contain a term, together with
//! how often the term appears in each one (term frequency, `tf`):
//!
//! ```text
//! "rust" -> [(doc 3, tf 2), (doc 7, tf 1), (doc 8, tf 5), (doc 120, tf 1)]
//! ```
//!
//! Doc ids are sorted, so instead of storing them directly we store the *gap*
//! from the previous id (delta encoding): `3, 4, 1, 112`. Gaps are small
//! numbers, and small numbers are cheap in varint encoding — most take a single
//! byte instead of four.
//!
//! On-disk layout of one list: `delta₀ tf₀ delta₁ tf₁ ...`, every number a varint.

use crate::varint;

/// One entry in a posting list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posting {
    pub doc: u32,
    pub tf: u32,
}

/// Encodes postings (which must be sorted by strictly increasing doc id) into `out`.
pub fn encode(postings: &[Posting], out: &mut Vec<u8>) {
    let mut prev = 0u32;
    for (i, p) in postings.iter().enumerate() {
        debug_assert!(
            i == 0 || p.doc > prev,
            "doc ids must be strictly increasing"
        );
        varint::encode(u64::from(p.doc - prev), out);
        varint::encode(u64::from(p.tf), out);
        prev = p.doc;
    }
}

/// Lazily decodes a posting list straight out of a byte slice (e.g. a memory map).
/// Nothing is allocated; each `next()` reads two varints.
pub struct PostingIter<'a> {
    data: &'a [u8],
    pos: usize,
    doc: u32,
    remaining: u32,
}

impl<'a> PostingIter<'a> {
    pub fn new(data: &'a [u8], doc_freq: u32) -> Self {
        Self {
            data,
            pos: 0,
            doc: 0,
            remaining: doc_freq,
        }
    }

    fn read(&mut self) -> Option<u32> {
        let (value, used) = varint::decode(&self.data[self.pos..])?;
        self.pos += used;
        u32::try_from(value).ok()
    }
}

impl Iterator for PostingIter<'_> {
    type Item = Posting;

    fn next(&mut self) -> Option<Posting> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let delta = self.read()?;
        let tf = self.read()?;
        self.doc += delta;
        Some(Posting { doc: self.doc, tf })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.remaining as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_size() {
        let postings: Vec<Posting> = [3, 7, 8, 120, 100_000]
            .iter()
            .map(|&doc| Posting {
                doc,
                tf: doc % 5 + 1,
            })
            .collect();
        let mut buf = Vec::new();
        encode(&postings, &mut buf);
        // gaps 3,4,1,112 fit in one byte each; 99_880 needs three. Every tf is one byte.
        assert_eq!(buf.len(), 4 + 3 + 5);
        let decoded: Vec<Posting> = PostingIter::new(&buf, postings.len() as u32).collect();
        assert_eq!(decoded, postings);
    }

    #[test]
    fn empty_list() {
        let mut buf = Vec::new();
        encode(&[], &mut buf);
        assert!(buf.is_empty());
        assert_eq!(PostingIter::new(&buf, 0).count(), 0);
    }
}
