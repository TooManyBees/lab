use super::{Lab, CBRT_EPSILON, EPSILON, KAPPA};
use std::arch::x86_64::*;
use std::{f32, iter, mem};

static BLANK_LAB: Lab = Lab {
    l: f32::NAN,
    a: f32::NAN,
    b: f32::NAN,
};

/// Converts a slice of `Lab`s to `[u8; 3]` RGB triples using 256-bit SIMD operations.
///
/// # Panics
/// This function will panic if executed on a non-x86_64 CPU or one without AVX
/// and SSE 4.1 support.
/// ```ignore
/// if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///     lab::simd::labs_to_rgbs(&labs);
/// }
/// ```
pub fn labs_to_rgbs(labs: &[Lab]) -> Vec<[u8; 3]> {
    let chunks = labs.chunks_exact(8);
    let remainder = chunks.remainder();
    let mut vs = chunks.fold(Vec::with_capacity(labs.len()), |mut v, labs| {
        let rgbs = unsafe { slice_labs_to_slice_rgbs(labs) };
        v.extend_from_slice(&rgbs);
        v
    });

    // While we could simplify this block by just calling the scalar version
    // of the code on the remainder, there are some variations between scalar
    // and SIMD floating point math (especially on TravisCI for some reason?)
    // and I don't want the trailing N items to be computed by a different
    // algorithm.
    if remainder.len() > 0 {
        let labs: Vec<Lab> = remainder
            .iter()
            .cloned()
            .chain(iter::repeat(BLANK_LAB))
            .take(8)
            .collect();

        let rgbs = unsafe { slice_labs_to_slice_rgbs(&labs) };
        vs.extend_from_slice(&rgbs[..remainder.len()]);
    }

    vs
}

/// Convert a slice of 8 `Lab` structs into an array of 8 RGB (`[u8; 3]`) triples.
///
/// This is the fundamental unit of work that `lab::simd::labs_to_rgbs` performs.
/// If you need to control how to parallelize this work, use this function.
///
/// Only the first 8 elements of the input slice will be converted. The example given
/// is very close to the implementation of `lab::simd::labs_to_rgbs`. Because this
/// library makes no assumptions about how to parallelize work, use this function
/// to add parallelization with Rayon, etc.
///
/// # Example
/// ```
/// # use lab::Lab;
/// # use std::{iter, f32};
/// # let labs: Vec<Lab> = {
/// #     let values: &[[f32; 3]] = &[[0.44953918, 0.2343294, 0.9811987], [0.66558355, 0.86746496, 0.6557031], [0.3853534, 0.5447681, 0.563337], [0.5060024, 0.002653122, 0.28564066], [0.112734795, 0.42281234, 0.5662596], [0.61263186, 0.7541826, 0.7710692], [0.35402274, 0.6711668, 0.090500355], [0.09291971, 0.18202633, 0.27621543], [0.74104124, 0.56239027, 0.6807165], [0.19430345, 0.46403062, 0.31903458], [0.9805223, 0.22615737, 0.6665648], [0.61051553, 0.66672426, 0.2612421]];
/// #     values.iter().map(|lab| lab::Lab { l: lab[0], a: lab[1], b: lab[2] }).collect()
/// # };
/// ##[cfg(target_arch = "x86_64")]
/// {
///     if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
///         let chunks = labs.chunks_exact(8);
///         let remainder = chunks.remainder();
///         // Parallelizing work with Rayon? Do it here, at `.fold()`
///         let mut vs = chunks.fold(Vec::with_capacity(labs.len()), |mut v, labs| {
///             let rgbs = lab::simd::labs_to_rgbs_chunk(labs);
///             v.extend_from_slice(&rgbs);
///             v
///         });
///
///         if remainder.len() > 0 {
///             const BLANK_LAB: Lab = Lab { l: f32::NAN, a: f32::NAN, b: f32::NAN };
///             let labs: Vec<Lab> =
///                 remainder.iter().cloned().chain(iter::repeat(BLANK_LAB))
///                 .take(8)
///                 .collect();
///
///             let rgbs = lab::simd::labs_to_rgbs_chunk(&labs);
///             vs.extend_from_slice(&rgbs[..remainder.len()]);
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
pub fn labs_to_rgbs_chunk(labs: &[Lab]) -> [[u8; 3]; 8] {
    unsafe { slice_labs_to_slice_rgbs(labs) }
}

#[inline]
unsafe fn slice_labs_to_slice_rgbs(labs: &[Lab]) -> [[u8; 3]; 8] {
    let (l, a, b) = lab_slice_to_simd(labs);
    let (x, y, z) = labs_to_xyzs(l, a, b);
    let (r, g, b) = xyzs_to_rgbs(x, y, z);
    simd_to_rgb_array(r, g, b)
}

#[inline]
unsafe fn lab_slice_to_simd(labs: &[Lab]) -> (__m256, __m256, __m256) {
    let labs = &labs[..8];
    let l = _mm256_set_ps(
        labs[0].l, labs[1].l, labs[2].l, labs[3].l, labs[4].l, labs[5].l, labs[6].l, labs[7].l,
    );
    let a = _mm256_set_ps(
        labs[0].a, labs[1].a, labs[2].a, labs[3].a, labs[4].a, labs[5].a, labs[6].a, labs[7].a,
    );
    let b = _mm256_set_ps(
        labs[0].b, labs[1].b, labs[2].b, labs[3].b, labs[4].b, labs[5].b, labs[6].b, labs[7].b,
    );
    (l, a, b)
}

#[inline]
unsafe fn labs_to_xyzs(l: __m256, a: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let fy = _mm256_div_ps(
        _mm256_add_ps(l, _mm256_set1_ps(16.0)),
        _mm256_set1_ps(116.0),
    );
    let fx = _mm256_add_ps(_mm256_div_ps(a, _mm256_set1_ps(500.0)), fy);
    let fz = _mm256_sub_ps(fy, _mm256_div_ps(b, _mm256_set1_ps(200.0)));

    let xr = {
        let false_branch = {
            let temp1 = _mm256_mul_ps(fx, _mm256_set1_ps(116.0));
            let temp2 = _mm256_sub_ps(temp1, _mm256_set1_ps(16.0));
            _mm256_div_ps(temp2, _mm256_set1_ps(KAPPA))
        };
        let unpacked_false_branch: [f32; 8] = mem::transmute(false_branch);
        let mut unpacked: [f32; 8] = mem::transmute(fx);
        for (i, el) in unpacked.iter_mut().enumerate() {
            if *el > CBRT_EPSILON {
                *el = el.powi(3);
            } else {
                *el = unpacked_false_branch[i];
            }
        }
        mem::transmute(unpacked)
    };

    let yr = {
        let false_branch = _mm256_div_ps(l, _mm256_set1_ps(KAPPA));
        let unpacked_false_branch: [f32; 8] = mem::transmute(false_branch);
        let unpacked_fy: [f32; 8] = mem::transmute(fy);
        let mut unpacked: [f32; 8] = mem::transmute(l);
        for (i, el) in unpacked.iter_mut().enumerate() {
            if *el > EPSILON * KAPPA {
                *el = unpacked_fy[i].powi(3);
            } else {
                *el = unpacked_false_branch[i];
            }
        }
        mem::transmute(unpacked)
    };

    let zr = {
        let false_branch = {
            let temp1 = _mm256_mul_ps(fz, _mm256_set1_ps(116.0));
            let temp2 = _mm256_sub_ps(temp1, _mm256_set1_ps(16.0));
            _mm256_div_ps(temp2, _mm256_set1_ps(KAPPA))
        };
        let unpacked_false_branch: [f32; 8] = mem::transmute(false_branch);
        let mut unpacked: [f32; 8] = mem::transmute(fz);
        for (i, el) in unpacked.iter_mut().enumerate() {
            if *el > CBRT_EPSILON {
                *el = el.powi(3);
            } else {
                *el = unpacked_false_branch[i];
            }
        }
        mem::transmute(unpacked)
    };

    (
        _mm256_mul_ps(xr, _mm256_set1_ps(0.95047)),
        yr,
        _mm256_mul_ps(zr, _mm256_set1_ps(1.08883)),
    )
}

#[inline]
unsafe fn xyzs_to_rgbs(x: __m256, y: __m256, z: __m256) -> (__m256, __m256, __m256) {
    let r = {
        let prod_x = _mm256_mul_ps(x, _mm256_set1_ps(3.2404541621141054));
        let prod_y = _mm256_mul_ps(y, _mm256_set1_ps(-1.5371385127977166));
        let prod_z = _mm256_mul_ps(z, _mm256_set1_ps(-0.4985314095560162));
        let sum = _mm256_add_ps(_mm256_add_ps(prod_x, prod_y), prod_z);
        xyzs_to_rgbs_map(sum)
    };
    let g = {
        let prod_x = _mm256_mul_ps(x, _mm256_set1_ps(-0.9692660305051868));
        let prod_y = _mm256_mul_ps(y, _mm256_set1_ps(1.8760108454466942));
        let prod_z = _mm256_mul_ps(z, _mm256_set1_ps(0.04155601753034984));
        let sum = _mm256_add_ps(_mm256_add_ps(prod_x, prod_y), prod_z);
        xyzs_to_rgbs_map(sum)
    };
    let b = {
        let prod_x = _mm256_mul_ps(x, _mm256_set1_ps(0.05564343095911469));
        let prod_y = _mm256_mul_ps(y, _mm256_set1_ps(-0.20402591351675387));
        let prod_z = _mm256_mul_ps(z, _mm256_set1_ps(1.0572251882231791));
        let sum = _mm256_add_ps(_mm256_add_ps(prod_x, prod_y), prod_z);
        xyzs_to_rgbs_map(sum)
    };

    (r, g, b)
}

#[inline]
unsafe fn xyzs_to_rgbs_map(c: __m256) -> __m256 {
    let mask = _mm256_cmp_ps(c, _mm256_set1_ps(0.0031308), _CMP_GT_OQ);
    let false_branch = _mm256_mul_ps(c, _mm256_set1_ps(12.92));
    let true_branch = {
        let mut unpacked: [f32; 8] = mem::transmute(c);
        let unpacked_mask: [f32; 8] = mem::transmute(mask);
        // Avoid `powf` at all costs by only operating on the elements that
        // will make it through the mask. `powf` is a significant performance hit
        for (el, test) in unpacked.iter_mut().zip(unpacked_mask.iter()) {
            if test.is_nan() {
                // NaN == true, 0.0 == false
                *el = el.powf(1.0 / 2.4);
            }
        }
        let raised: __m256 = mem::transmute(unpacked);
        let temp2 = _mm256_mul_ps(raised, _mm256_set1_ps(1.055));
        _mm256_sub_ps(temp2, _mm256_set1_ps(0.055))
    };
    let blended = _mm256_blendv_ps(false_branch, true_branch, mask);
    _mm256_mul_ps(blended, _mm256_set1_ps(255.0))
}

#[inline]
unsafe fn simd_to_rgb_array(r: __m256, g: __m256, b: __m256) -> [[u8; 3]; 8] {
    let r: [f32; 8] = mem::transmute(_mm256_round_ps(r, _MM_FROUND_TO_NEAREST_INT));
    let g: [f32; 8] = mem::transmute(_mm256_round_ps(g, _MM_FROUND_TO_NEAREST_INT));
    let b: [f32; 8] = mem::transmute(_mm256_round_ps(b, _MM_FROUND_TO_NEAREST_INT));

    let mut rgbs: [[u8; 3]; 8] = mem::uninitialized();
    for (((&r, &g), &b), rgb) in r
        .iter()
        .zip(g.iter())
        .zip(b.iter())
        .rev()
        .zip(rgbs.iter_mut())
    {
        *rgb = [r as u8, g as u8, b as u8];
    }
    rgbs
}

// #[cfg(all(target_cpu = "x86_64", target_feature = "avx", target_feature = "sse4.1"))]
#[cfg(test)]
mod test {
    use super::super::super::{labs_to_rgbs, simd, Lab};
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
    fn test_simd_labs_to_rgbs() {
        let labs = simd::rgbs_to_labs(&RGBS);
        let rgbs = simd::labs_to_rgbs(&labs);
        assert_eq!(rgbs.as_slice(), RGBS.as_slice());
    }

    #[test]
    fn test_simd_labs_to_rgbs_unsaturated() {
        let labs = vec![Lab {
            l: 66.6348,
            a: 52.260696,
            b: 14.850557,
        }];
        let rgbs_non_simd = labs_to_rgbs(&labs);
        let rgbs_simd = simd::labs_to_rgbs(&labs);
        assert_eq!(rgbs_simd, rgbs_non_simd);
    }
}
