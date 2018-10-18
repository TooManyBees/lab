use std::arch::x86_64::*;
use std::{iter, mem, f32};
use super::{Lab, KAPPA, EPSILON, CBRT_EPSILON};

static BLANK_LAB: Lab = Lab { l: f32::NAN, a: f32::NAN, b: f32::NAN };

pub unsafe fn labs_to_rgbs(labs: &[Lab]) -> Vec<[u8; 3]> {
    labs.chunks(8).fold(Vec::with_capacity(labs.len()), |mut v, labs| {
        let rgbs: Vec<_> = match labs {
            packed @ &[_, _, _, _, _, _, _, _] => {
                vec_labs_to_vec_rgbs(packed)
            },
            rest => {
                let labs: Vec<Lab> =
                    rest.iter().cloned().chain(iter::repeat(BLANK_LAB))
                    .take(8)
                    .collect();

                let mut rgbs = vec_labs_to_vec_rgbs(&labs);
                rgbs.truncate(rest.len());
                rgbs
            },
        };
        v.extend_from_slice(&rgbs);
        v
    })
}

unsafe fn vec_labs_to_vec_rgbs(labs: &[Lab]) -> Vec<[u8; 3]> {
    let (l, a, b) = lab_slice_to_simd(labs);
    let (x, y, z) = labs_to_xyzs(l, a, b);
    let (r, g, b) = xyzs_to_rgbs(x, y, z);
    simd_to_rgb_vec(r, g, b)
}

#[inline]
unsafe fn lab_slice_to_simd(labs: &[Lab]) -> (__m256, __m256, __m256) {
    let labs = &labs[..8];
    let l = _mm256_set_ps(labs[0].l, labs[1].l, labs[2].l, labs[3].l, labs[4].l, labs[5].l, labs[6].l, labs[7].l);
    let a = _mm256_set_ps(labs[0].a, labs[1].a, labs[2].a, labs[3].a, labs[4].a, labs[5].a, labs[6].a, labs[7].a);
    let b = _mm256_set_ps(labs[0].b, labs[1].b, labs[2].b, labs[3].b, labs[4].b, labs[5].b, labs[6].b, labs[7].b);
    (l, a, b)
}

#[inline]
unsafe fn labs_to_xyzs(l: __m256, a: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let fy = _mm256_div_ps(_mm256_add_ps(l, _mm256_set1_ps(16.0)), _mm256_set1_ps(116.0));
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
unsafe fn xyzs_to_rgbs_map(c: __m256) ->  __m256 {
    let mask = _mm256_cmp_ps(c, _mm256_set1_ps(0.0031308), _CMP_GT_OQ);
    let false_branch = _mm256_mul_ps(c, _mm256_set1_ps(12.92));
    let true_branch = {
        let mut unpacked: [f32; 8] = mem::transmute(c);
        let unpacked_mask: [f32; 8] = mem::transmute(mask);
        // Avoid `powf` at all costs by only operating on the elements that
        // will make it through the mask. `powf` is a significant performance hit
        for (el, test) in unpacked.iter_mut().zip(unpacked_mask.iter()) {
            if test.is_nan() { // NaN == true, 0.0 == false
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
unsafe fn simd_to_rgb_vec(r: __m256, g: __m256, b: __m256) -> Vec<[u8; 3]> {
    let r: [f32; 8] = mem::transmute(_mm256_round_ps(r, _MM_FROUND_TO_NEAREST_INT));
    let g: [f32; 8] = mem::transmute(_mm256_round_ps(g, _MM_FROUND_TO_NEAREST_INT));
    let b: [f32; 8] = mem::transmute(_mm256_round_ps(b, _MM_FROUND_TO_NEAREST_INT));
    r.iter().zip(g.iter()).zip(b.iter()).map(|((&r, &g), &b)| [r as u8, g as u8, b as u8]).rev().collect()
}

// #[cfg(all(target_cpu = "x86_64", target_feature = "avx", target_feature = "sse4.1"))]
#[cfg(test)]
mod test {
    use rand;
    use rand::Rng;
    use rand::distributions::Standard;
    use super::super::super::{Lab, labs_to_rgbs, simd};

    lazy_static! {
        static ref RGBS: Vec<[u8;3]> = {
            let rand_seed = [0u8;32];
            let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
            rng.sample_iter(&Standard).take(512).collect()
        };
    }

    #[test]
    fn test_simd_labs_to_rgbs() {
        let labs = unsafe { simd::rgbs_to_labs(&RGBS) };
        let rgbs = unsafe { simd::labs_to_rgbs(&labs) };
        assert_eq!(rgbs.as_slice(), RGBS.as_slice());
    }

    #[test]
    fn test_simd_labs_to_rgbs_unsaturated() {
        let labs = vec![
            Lab { l: 66.6348, a: 52.260696, b: 14.850557 },
        ];
        let rgbs_non_simd = labs_to_rgbs(&labs);
        let rgbs_simd = unsafe { simd::labs_to_rgbs(&labs) };
        assert_eq!(rgbs_simd, rgbs_non_simd);
    }
}
