//! Exact nearest-neighbor search by brute force: compare the query with every
//! vector and keep the best `k`.
//!
//! O(n·d) per query. It is the baseline everything else is measured against:
//! always correct (recall 1.0), and for a few thousand vectors honestly fast
//! enough — 10k × 384 dims is ~4M multiply-adds, around a millisecond with SIMD.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::Neighbor;
use crate::distance;
use crate::vectors::VectorSource;

/// Orders by score, then id, so heap behavior is deterministic.
#[derive(PartialEq)]
struct ByScore(f32, u32);

impl Eq for ByScore {}
impl PartialOrd for ByScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(other.1.cmp(&self.1))
    }
}

/// The `k` vectors most similar to `query` (which should be unit length).
pub fn search(vectors: &impl VectorSource, query: &[f32], k: usize) -> Vec<Neighbor> {
    if k == 0 {
        return Vec::new();
    }
    // A min-heap of the best k so far: its top is the *worst* of the best,
    // the one to evict when something better shows up. O(n log k).
    let mut heap: BinaryHeap<std::cmp::Reverse<ByScore>> = BinaryHeap::with_capacity(k + 1);
    for id in 0..vectors.len() as u32 {
        let score = distance::dot(query, vectors.get(id));
        if heap.len() < k {
            heap.push(std::cmp::Reverse(ByScore(score, id)));
        } else if score > heap.peek().unwrap().0.0 {
            heap.pop();
            heap.push(std::cmp::Reverse(ByScore(score, id)));
        }
    }
    let mut out: Vec<Neighbor> = heap
        .into_iter()
        .map(|r| Neighbor {
            id: r.0.1,
            score: r.0.0,
        })
        .collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectors::VecStore;

    #[test]
    fn finds_most_similar() {
        let mut store = VecStore::new(2);
        store.push(&[1.0, 0.0]); // 0
        store.push(&[0.0, 1.0]); // 1
        store.push(&[1.0, 1.0]); // 2
        store.push(&[-1.0, 0.0]); // 3
        let hits = search(&store, &[1.0, 0.1], 2);
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), [0, 2]);
        assert!(hits[0].score > hits[1].score);
        assert_eq!(search(&store, &[1.0, 0.0], 10).len(), 4);
    }
}
