//! Flat `f32` vector storage: every vector back to back in one contiguous array.
//!
//! ```text
//! data: [v0[0] v0[1] … v0[d-1] | v1[0] … v1[d-1] | v2[0] …]
//! ```
//!
//! Vector `i` is `data[i*d .. (i+1)*d]`: no pointers to chase, and a scan
//! over all vectors reads memory strictly front to back, which is what CPU
//! prefetchers and SIMD like best.

use crate::distance;

/// Anything that hands out vectors by id: in-memory while building,
/// memory-mapped when serving.
pub trait VectorSource: Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn get(&self, id: u32) -> &[f32];

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct VecStore {
    dim: usize,
    data: Vec<f32>,
}

impl VecStore {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0);
        Self {
            dim,
            data: Vec::new(),
        }
    }

    /// Adds a vector (normalized to unit length) and returns its id.
    pub fn push(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim, "vector has the wrong dimension");
        let id = self.len() as u32;
        let start = self.data.len();
        self.data.extend_from_slice(v);
        distance::normalize(&mut self.data[start..]);
        id
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

impl VectorSource for VecStore {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    fn get(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
    }
}
