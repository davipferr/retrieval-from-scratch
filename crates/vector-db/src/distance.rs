//! Vector similarity, three ways.
//!
//! All vectors in this crate are L2-normalized when they are added, so cosine
//! similarity is just the dot product: `cos(a, b) = a·b / (|a||b|) = a·b`.
//! The dot product is therefore the hot loop of every search, and is written
//! three times:
//!
//! 1. [`dot_scalar`]: the obvious loop. The compiler will *not* vectorize it,
//!    because float addition isn't associative and reordering the sum would
//!    change the result. It must add the elements one at a time, in order.
//! 2. [`dot_wide`]: portable SIMD with the `wide` crate. Eight independent
//!    lanes (`f32x8`), each summing every 8th element, combined at the end.
//!    We're explicitly allowing the reordering the compiler couldn't assume.
//! 3. [`dot_avx2`]: the same idea with raw `std::arch` intrinsics, using fused
//!    multiply-add and two accumulators to hide FMA latency. Picked at runtime
//!    only if the CPU supports AVX2 + FMA.
//!
//! [`dot`] dispatches to the fastest one available, decided once.

use std::sync::LazyLock;

use wide::f32x8;

pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn dot_wide(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = f32x8::ZERO;
    // [f32; 8] blocks plus a remainder of up to 7
    let ((a_blocks, a_tail), (b_blocks, b_tail)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (&x, &y) in a_blocks.iter().zip(b_blocks) {
        acc = f32x8::from(x).mul_add(f32x8::from(y), acc);
    }
    let tail = dot_scalar(a_tail, b_tail);
    acc.reduce_add() + tail
}

/// AVX2 + FMA. Returns `None` on CPUs (or architectures) without them.
pub fn dot_avx2(a: &[f32], b: &[f32]) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        assert_eq!(a.len(), b.len());
        // SAFETY: we just checked that the CPU supports both features.
        return Some(unsafe { x86::dot_avx2_fma(a, b) });
    }
    let _ = (a, b);
    None
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        // SAFETY: every load reads 8 floats starting at i < n - 15 (or n - 7),
        // so it stays inside both slices, which have equal length.
        unsafe {
            // Two accumulators: an FMA takes ~4 cycles but the CPU can start
            // two per cycle, so one dependency chain would leave it idle.
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= n {
                acc0 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 8)),
                    _mm256_loadu_ps(pb.add(i + 8)),
                    acc1,
                );
                i += 16;
            }
            if i + 8 <= n {
                acc0 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
                i += 8;
            }
            // horizontal sum of the 8 lanes: 8 -> 4 -> 2 -> 1
            let acc = _mm256_add_ps(acc0, acc1);
            let quad = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
            let single = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1));
            let mut sum = _mm_cvtss_f32(single);
            while i < n {
                sum += a[i] * b[i];
                i += 1;
            }
            sum
        }
    }
}

type DotFn = fn(&[f32], &[f32]) -> f32;

static BEST_DOT: LazyLock<(DotFn, &'static str)> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let avx2: DotFn = |a, b| {
            assert_eq!(a.len(), b.len());
            // SAFETY: this function is only selected after the feature check above.
            unsafe { x86::dot_avx2_fma(a, b) }
        };
        return (avx2, "avx2+fma");
    }
    (dot_wide, "wide f32x8")
});

/// Dot product using the fastest implementation this CPU supports.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    (BEST_DOT.0)(a, b)
}

/// Name of the implementation [`dot`] uses.
pub fn dot_impl_name() -> &'static str {
    BEST_DOT.1
}

/// HNSW works with distances (smaller = closer); for unit vectors this is
/// `1 - cos`, in `[0, 2]`.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

/// Scales `v` to unit length (leaves an all-zero vector alone).
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vectors(len: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
        let b = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
        (a, b)
    }

    #[test]
    fn implementations_agree() {
        // lengths around the 8- and 16-wide boundaries exercise every tail path
        for len in [0, 1, 7, 8, 9, 15, 16, 17, 31, 384, 385] {
            let (a, b) = vectors(len);
            let expected = dot_scalar(&a, &b);
            let tol = 1e-4 * (1.0 + expected.abs());
            assert!((dot_wide(&a, &b) - expected).abs() < tol, "wide, len {len}");
            if let Some(v) = dot_avx2(&a, &b) {
                assert!((v - expected).abs() < tol, "avx2, len {len}");
            }
            assert!((dot(&a, &b) - expected).abs() < tol, "dispatch, len {len}");
        }
    }

    #[test]
    fn normalized_cosine() {
        let mut a = vec![3.0, 4.0];
        normalize(&mut a);
        assert!((a[0] - 0.6).abs() < 1e-6 && (a[1] - 0.8).abs() < 1e-6);
        assert!(cosine_distance(&a, &a).abs() < 1e-6);
        let mut b = vec![-3.0, -4.0];
        normalize(&mut b);
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-6);
    }
}
