use crate::{Lab, EPSILON, KAPPA};
use crate::simd::math::powf256_ps;
use std::arch::x86_64::*;
use std::{iter, mem};

static BLANK_RGB: [u8; 3] = [0u8; 3];

/// Converts a slice of `[u8; 3]` RGB triples to `Lab`s using 256-bit SIMD operations.
///
/// # Panics
/// This function will panic if executed on a non-x86_64 CPU or one without AVX
/// and SSE 4.1 support.
/// ```ignore
/// if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///     lab::simd::rgbs_to_labs(&rgbs);
/// }
/// ```
pub fn rgbs_to_labs(rgbs: &[[u8; 3]]) -> Vec<Lab> {
    let chunks = rgbs.chunks_exact(8);
    let remainder = chunks.remainder();
    let mut vs = chunks.fold(Vec::with_capacity(rgbs.len()), |mut v, rgbs| {
        let labs = unsafe { slice_rgbs_to_slice_labs(rgbs) };
        v.extend_from_slice(&labs);
        v
    });

    // While we could simplify this block by just calling the scalar version
    // of the code on the remainder, there are some variations between scalar
    // and SIMD floating point math (especially on TravisCI for some reason?)
    // and I don't want the trailing N items to be computed by a different
    // algorithm.
    if remainder.len() > 0 {
        let rgbs: Vec<[u8; 3]> = remainder
            .iter()
            .cloned()
            .chain(iter::repeat(BLANK_RGB))
            .take(8)
            .collect();
        let labs = unsafe { slice_rgbs_to_slice_labs(&rgbs) };
        vs.extend_from_slice(&labs[..remainder.len()]);
    }

    vs
}

/// Convert a slice of 8 `[u8; 3]` RGB tripes into an array of 8 `Lab` structs.
///
/// This is the fundamental unit of work that `lab::simd::rgbs_to_labs` performs.
/// If you need to control how to parallelize this work, use this function.
///
/// Only the first 8 elements of the input slice will be converted. The example given
/// is very close to the implementation of `lab::simd::rgbs_to_labs`. Because this
/// library makes no assumptions about how to parallelize work, use this function
/// to add parallelization with Rayon, etc.
///
/// # Example
/// ```
/// # use lab::Lab;
/// # use std::iter;
/// # let rgbs: Vec<[u8; 3]> = vec![];
/// ##[cfg(target_arch = "x86_64")]
/// {
///     if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///         let chunks = rgbs.chunks_exact(8);
///         let remainder = chunks.remainder();
///         // Parallelizing work with Rayon? Do it here, at `.fold()`
///         let mut vs = chunks.fold(Vec::with_capacity(rgbs.len()), |mut v, rgbs| {
///             let labs = lab::simd::rgbs_to_labs_chunk(rgbs);
///             v.extend_from_slice(&labs);
///             v
///         });
///
///         if remainder.len() > 0 {
///             const BLANK_RGB: [u8; 3] = [0u8; 3];
///             let rgbs: Vec<[u8; 3]> =
///                 remainder.iter().cloned().chain(iter::repeat(BLANK_RGB))
///                 .take(8)
///                 .collect();
///
///             let labs = lab::simd::rgbs_to_labs_chunk(&rgbs);
///             vs.extend_from_slice(&labs[..remainder.len()]);
///         }
///     }
/// }
/// ```
///
/// # Panics
/// This function will panic of the input slice has fewer than 8 elements. Consider
/// padding the input slice with blank values and then truncating the result.
///
/// Additionally, it will panic if run on a CPU that does not support x86_64's AVX
/// and SSE 4.1 instructions.
pub fn rgbs_to_labs_chunk(rgbs: &[[u8; 3]]) -> [Lab; 8] {
    unsafe { slice_rgbs_to_slice_labs(rgbs) }
}

unsafe fn slice_rgbs_to_slice_labs(rgbs: &[[u8; 3]]) -> [Lab; 8] {
    let (r, g, b) = rgb_slice_to_simd(rgbs);
    let (x, y, z) = rgbs_to_xyzs(r, g, b);
    let (l, a, b) = xyzs_to_labs(x, y, z);
    simd_to_lab_array(l, a, b)
}

#[inline]
unsafe fn rgb_slice_to_simd(rgbs: &[[u8; 3]]) -> (__m256, __m256, __m256) {
    let r = _mm256_set_ps(
        rgbs[0][0] as f32,
        rgbs[1][0] as f32,
        rgbs[2][0] as f32,
        rgbs[3][0] as f32,
        rgbs[4][0] as f32,
        rgbs[5][0] as f32,
        rgbs[6][0] as f32,
        rgbs[7][0] as f32,
    );
    let g = _mm256_set_ps(
        rgbs[0][1] as f32,
        rgbs[1][1] as f32,
        rgbs[2][1] as f32,
        rgbs[3][1] as f32,
        rgbs[4][1] as f32,
        rgbs[5][1] as f32,
        rgbs[6][1] as f32,
        rgbs[7][1] as f32,
    );
    let b = _mm256_set_ps(
        rgbs[0][2] as f32,
        rgbs[1][2] as f32,
        rgbs[2][2] as f32,
        rgbs[3][2] as f32,
        rgbs[4][2] as f32,
        rgbs[5][2] as f32,
        rgbs[6][2] as f32,
        rgbs[7][2] as f32,
    );
    (r, g, b)
}

unsafe fn rgbs_to_xyzs(r: __m256, g: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let r = rgbs_to_xyzs_map(r);
    let g = rgbs_to_xyzs_map(g);
    let b = rgbs_to_xyzs_map(b);

    let x = {
        let prod_r = _mm256_mul_ps(r, _mm256_set1_ps(0.4124564390896921));
        let prod_g = _mm256_mul_ps(g, _mm256_set1_ps(0.357576077643909));
        let prod_b = _mm256_mul_ps(b, _mm256_set1_ps(0.18043748326639894));
        _mm256_add_ps(_mm256_add_ps(prod_r, prod_g), prod_b)
    };

    let y = {
        let prod_r = _mm256_mul_ps(r, _mm256_set1_ps(0.21267285140562248));
        let prod_g = _mm256_mul_ps(g, _mm256_set1_ps(0.715152155287818));
        let prod_b = _mm256_mul_ps(b, _mm256_set1_ps(0.07217499330655958));
        _mm256_add_ps(_mm256_add_ps(prod_r, prod_g), prod_b)
    };

    let z = {
        let prod_r = _mm256_mul_ps(r, _mm256_set1_ps(0.019333895582329317));
        let prod_g = _mm256_mul_ps(g, _mm256_set1_ps(0.119192025881303));
        let prod_b = _mm256_mul_ps(b, _mm256_set1_ps(0.9503040785363677));
        _mm256_add_ps(_mm256_add_ps(prod_r, prod_g), prod_b)
    };

    (x, y, z)
}

#[inline]
unsafe fn rgbs_to_xyzs_map(c: __m256) -> __m256 {
    let mask = _mm256_cmp_ps(c, _mm256_set1_ps(10.0), _CMP_GT_OQ);
    let true_branch = {
        const A: f32 = 0.055 * 255.0;
        const D: f32 = 1.055 * 255.0;
        let t0 = _mm256_div_ps(_mm256_add_ps(c, _mm256_set1_ps(A)), _mm256_set1_ps(D));
        powf256_ps(t0, _mm256_set1_ps(2.4))
    };

    let false_branch = {
        const D: f32 = 12.92 * 255.0;
        _mm256_div_ps(c, _mm256_set1_ps(D))
    };
    _mm256_blendv_ps(false_branch, true_branch, mask)
}

unsafe fn xyzs_to_labs(x: __m256, y: __m256, z: __m256) -> (__m256, __m256, __m256) {
    let x = xyzs_to_labs_map(_mm256_div_ps(x, _mm256_set1_ps(0.95047)));
    let y = xyzs_to_labs_map(y);
    let z = xyzs_to_labs_map(_mm256_div_ps(z, _mm256_set1_ps(1.08883)));

    let l = _mm256_add_ps(
        _mm256_mul_ps(y, _mm256_set1_ps(116.0)),
        _mm256_set1_ps(-16.0),
    );
    let a = _mm256_mul_ps(_mm256_sub_ps(x, y), _mm256_set1_ps(500.0));
    let b = _mm256_mul_ps(_mm256_sub_ps(y, z), _mm256_set1_ps(200.0));

    (l, a, b)
}

#[inline]
unsafe fn xyzs_to_labs_map(c: __m256) -> __m256 {
    let mask = _mm256_cmp_ps(c, _mm256_set1_ps(EPSILON), _CMP_GT_OQ);
    // do false branch first
    let false_branch = _mm256_div_ps(
        _mm256_add_ps(
            _mm256_mul_ps(c, _mm256_set1_ps(KAPPA)),
            _mm256_set1_ps(16.0),
        ),
        _mm256_set1_ps(116.0),
    );
    let true_branch = powf256_ps(c, _mm256_set1_ps(1.0 / 3.0));
    _mm256_blendv_ps(false_branch, true_branch, mask)
}

unsafe fn simd_to_lab_array(l: __m256, a: __m256, b: __m256) -> [Lab; 8] {
    let l: [f32; 8] = mem::transmute(l);
    let a: [f32; 8] = mem::transmute(a);
    let b: [f32; 8] = mem::transmute(b);

    // Per `https://doc.rust-lang.org/std/mem/union.MaybeUninit.html` this `assume_init`
    // is safe because the MaybeUninits inside the array don't require initialization?
    let mut labs: [mem::MaybeUninit<Lab>; 8] = mem::MaybeUninit::uninit().assume_init();
    for (((&l, &a), &b), lab) in l
        .iter()
        .zip(a.iter())
        .zip(b.iter())
        .rev()
        .zip(labs.iter_mut())
    {
        *lab = mem::MaybeUninit::new(Lab { l, a, b });
    }
    mem::transmute(labs)
}

// #[inline]
// unsafe fn clamp(r: __m256, g: __m256, b: __m256) -> (__m256, __m256, __m256) {
//     let max = _mm256_set1_ps(1.0);
//     let min = _mm256_set1_ps(0.0);
//     let r = _mm256_max_ps(_mm256_min_ps(r, max), min);
//     let g = _mm256_max_ps(_mm256_min_ps(g, max), min);
//     let b = _mm256_max_ps(_mm256_min_ps(b, max), min);
//     (r, g, b)
// }

// #[inline]
// unsafe fn normalize_short_to_unit(r: __m256i, g: __m256i, b: __m256i) -> (__m256, __m256, __m256) {
//     let normalizer = _mm256_set1_ps(255.0);
//     let r = _mm256_div_ps(r, normalizer);
//     let g = _mm256_div_ps(g, normalizer);
//     let b = _mm256_div_ps(b, normalizer);
//     (r, g, b)
// }

// #[cfg(all(target_cpu = "x86_64", target_feature = "avx", target_feature = "sse4.1"))]
#[cfg(test)]
mod test {
    use super::super::super::{rgbs_to_labs, simd};
    use rand;
    use rand::distributions::Standard;
    use rand::Rng;

    lazy_static! {
        static ref RGBS: Vec<[u8; 3]> = {
            let rand_seed = [0u8; 32];
            let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
            rng.sample_iter(&Standard).take(512).collect()
        };
    }

    #[test]
    fn test_simd_rgbs_to_labs() {
        let rgbs = vec![
            [253, 120, 138], // Lab { l: 66.6348, a: 52.260696, b: 14.850557 }
            [25, 20, 22],    // Lab { l: 6.9093895, a: 2.8204322, b: -0.45616925 }
            [63, 81, 181],   // Lab { l: 38.336494, a: 25.586218, b: -55.288517 }
            [21, 132, 102],  // Lab { l: 49.033485, a: -36.959187, b: 7.9363704 }
            [255, 193, 7],   // Lab { l: 81.519325, a: 9.4045105, b: 82.69791 }
            [233, 30, 99],   // Lab { l: 50.865776, a: 74.61989, b: 15.343171 }
            [155, 96, 132],  // Lab { l: 48.260345, a: 29.383003, b: -9.950054 }
            [249, 165, 33],  // Lab { l: 74.29188, a: 21.827251, b: 72.75864 }
        ];

        let labs_non_simd = rgbs_to_labs(&rgbs);
        let labs_simd = simd::rgbs_to_labs(&rgbs);
        assert_relative_eq!(labs_simd.as_slice(), labs_non_simd.as_slice(), max_relative = 0.00002);
    }

    #[test]
    fn test_simd_rgbs_to_labs_many() {
        let labs_non_simd = rgbs_to_labs(&RGBS);
        let labs_simd = simd::rgbs_to_labs(&RGBS);
        assert_relative_eq!(labs_simd.as_slice(), labs_non_simd.as_slice(), max_relative = 0.00005);
    }

    #[test]
    fn test_simd_rgbs_to_labs_unsaturated() {
        let rgbs = vec![[253, 120, 138]];
        let labs_non_simd = rgbs_to_labs(&rgbs);
        let labs_simd = simd::rgbs_to_labs(&rgbs);
        assert_relative_eq!(labs_simd.as_slice(), labs_non_simd.as_slice(), max_relative = 0.00002);
    }
}
