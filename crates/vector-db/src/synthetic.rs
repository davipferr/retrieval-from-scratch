//! Random test data for benchmarks and tests.

use crate::hnsw::Rng;
use crate::vectors::VecStore;

/// `n` unit vectors scattered around 20 random centers.
///
/// Real embeddings are clustered (texts about the same topic land close
/// together), which makes them much easier for HNSW than uniform noise;
/// clustered data gives more realistic recall numbers.
pub fn clustered(n: usize, dim: usize, seed: u64) -> VecStore {
    let mut rng = Rng(seed);
    let mut gauss = move || {
        // Box–Muller transform: two uniforms -> one standard normal
        let (u1, u2) = (rng.next_f64(), rng.next_f64());
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    };
    let centers: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..dim).map(|_| gauss()).collect())
        .collect();
    let mut store = VecStore::new(dim);
    for i in 0..n {
        let center = &centers[i % centers.len()];
        let v: Vec<f32> = center.iter().map(|x| x + 0.5 * gauss()).collect();
        store.push(&v);
    }
    store
}
