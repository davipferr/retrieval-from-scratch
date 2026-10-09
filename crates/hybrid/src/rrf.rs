//! Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).
//!
//! BM25 and vector search score on incompatible scales (BM25: 0 to ~30,
//! unbounded; cosine: −1 to 1), so their scores can't simply be added. RRF
//! ignores scores and uses only *ranks*:
//!
//! ```text
//! rrf(d) = Σ over lists   1 / (k + rank of d in that list)
//! ```
//!
//! A document ranked well by both lists beats one ranked first by only one.
//! `k` (60 in the paper) damps the head of each list, so being #1 instead of
//! #2 matters less than appearing in both lists at all.

use std::collections::HashMap;

pub const DEFAULT_K: f32 = 60.0;

/// Fuses ranked lists of ids (best first) into one, best first.
pub fn fuse(lists: &[&[u32]], k: f32) -> Vec<(u32, f32)> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for list in lists {
        for (rank, &id) in list.iter().enumerate() {
            *scores.entry(id).or_default() += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut fused: Vec<(u32, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agreement_beats_a_single_first_place() {
        let bm25 = [1, 2, 3];
        let vector = [4, 2, 5];
        let fused = fuse(&[&bm25, &vector], DEFAULT_K);
        assert_eq!(fused[0].0, 2, "2 is in both lists");
        let ids: Vec<u32> = fused.iter().map(|f| f.0).collect();
        assert_eq!(ids.len(), 5);
        assert!(ids.contains(&1) && ids.contains(&4));
    }

    #[test]
    fn single_list_keeps_its_order() {
        let fused = fuse(&[&[9, 3, 7]], DEFAULT_K);
        assert_eq!(fused.iter().map(|f| f.0).collect::<Vec<_>>(), [9, 3, 7]);
    }
}
