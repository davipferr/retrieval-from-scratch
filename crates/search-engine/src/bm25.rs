//! Okapi BM25, the default ranking function of Lucene/Elasticsearch since 2016.
//!
//! For a query term `t` and a document `d`:
//!
//! ```text
//!                           tf · (k1 + 1)
//! score(t, d) = idf(t) · ──────────────────────────────────
//!                         tf + k1 · (1 − b + b · |d| / avgdl)
//!
//! idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
//! ```
//!
//! - `idf`: rare terms matter more. "the" is in every doc and tells us nothing;
//!   "levenshtein" is in three and tells us a lot.
//! - `tf` saturation: the 10th occurrence of a word adds much less than the
//!   2nd. `k1` controls how fast the curve flattens (higher = slower).
//! - length normalization: a match in a short doc counts more than the same
//!   match buried in a long one. `b` = 0 disables it, 1 applies it fully.
//!
//! A document's score for a query is the sum over the query terms.

#[derive(Debug, Clone, Copy)]
pub struct Bm25 {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25 {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25 {
    /// The "+1" inside the log keeps idf positive even for terms in over half
    /// the documents (Lucene's variant; the original formula goes negative).
    pub fn idf(n_docs: usize, doc_freq: u32) -> f32 {
        let n = n_docs as f32;
        let df = doc_freq as f32;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    pub fn score(&self, idf: f32, tf: u32, doc_len: u32, avg_doc_len: f32) -> f32 {
        let tf = tf as f32;
        let norm = if avg_doc_len > 0.0 {
            doc_len as f32 / avg_doc_len
        } else {
            1.0
        };
        idf * tf * (self.k1 + 1.0) / (tf + self.k1 * (1.0 - self.b + self.b * norm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rare_terms_weigh_more() {
        assert!(Bm25::idf(1000, 3) > Bm25::idf(1000, 300));
        assert!(Bm25::idf(1000, 1000) > 0.0);
    }

    #[test]
    fn tf_saturates() {
        let bm = Bm25::default();
        let s1 = bm.score(1.0, 1, 100, 100.0);
        let s2 = bm.score(1.0, 2, 100, 100.0);
        let s20 = bm.score(1.0, 20, 100, 100.0);
        assert!(s2 > s1 && s20 > s2);
        assert!(s2 - s1 > (s20 - s2) / 18.0, "later occurrences add less");
        assert!(s20 < bm.k1 + 1.0, "bounded by k1 + 1");
    }

    #[test]
    fn shorter_docs_win() {
        let bm = Bm25::default();
        assert!(bm.score(1.0, 3, 50, 100.0) > bm.score(1.0, 3, 500, 100.0));
    }
}
