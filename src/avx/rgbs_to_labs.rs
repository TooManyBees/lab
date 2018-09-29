use std::arch::x86_64::*;
use std::{mem, f32};
use super::{Lab, KAPPA, EPSILON};

#[allow(dead_code)]
pub unsafe fn rgbs_to_labs(rgbs: &[[u8; 3]]) -> Vec<Lab> {
    rgbs.chunks(8).fold(Vec::with_capacity(rgbs.len()), |mut v, rgbs| {
        let labs = match rgbs {
            rgbs @ &[_, _, _, _, _, _, _, _] => {
                let (r, g, b) = rgb_slice_to_simd(rgbs);
                let (r, g, b) = uint_to_f32(r, g, b);
                let (x, y, z) = rgbs_to_xyzs(r, g, b);
                let (l, a, b) = xyzs_to_labs(x, y, z);
                simd_to_lab_vec(l, a, b)
            },
            rest => {
                let mut rgbs: Vec<[f32; 3]> = Vec::with_capacity(8);
                for rgb in rest.iter() {
                    rgbs.push([rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]);
                }
                let num_padding = 8 - rest.len();
                for _ in 0..num_padding {
                    rgbs.push([f32::NAN; 3]);
                }
                let (r, g, b) = {
                    let rgbs = &rgbs[..8];
                    (
                        _mm256_set_ps(rgbs[0][0], rgbs[1][0], rgbs[2][0], rgbs[3][0], rgbs[4][0], rgbs[5][0], rgbs[6][0], rgbs[7][0]),
                        _mm256_set_ps(rgbs[0][1], rgbs[1][1], rgbs[2][1], rgbs[3][1], rgbs[4][1], rgbs[5][1], rgbs[6][1], rgbs[7][1]),
                        _mm256_set_ps(rgbs[0][2], rgbs[1][2], rgbs[2][2], rgbs[3][2], rgbs[4][2], rgbs[5][2], rgbs[6][2], rgbs[7][2]),
                    )
                };
                let (x, y, z) = rgbs_to_xyzs(r, g, b);
                let (l, a, b) = xyzs_to_labs(x, y, z);
                let mut labs = simd_to_lab_vec(l, a, b);
                labs.truncate(rest.len());
                labs
            },
        };
        v.extend_from_slice(&labs);
        v
    })
}

#[inline]
unsafe fn rgb_slice_to_simd(rgbs: &[[u8; 3]]) -> (__m256i, __m256i, __m256i) {
    let rgbs = &rgbs[..8];
    let r = _mm256_set_epi32(rgbs[0][0] as i32, rgbs[1][0] as i32, rgbs[2][0] as i32, rgbs[3][0] as i32, rgbs[4][0] as i32, rgbs[5][0] as i32, rgbs[6][0] as i32, rgbs[7][0] as i32);
    let g = _mm256_set_epi32(rgbs[0][1] as i32, rgbs[1][1] as i32, rgbs[2][1] as i32, rgbs[3][1] as i32, rgbs[4][1] as i32, rgbs[5][1] as i32, rgbs[6][1] as i32, rgbs[7][1] as i32);
    let b = _mm256_set_epi32(rgbs[0][2] as i32, rgbs[1][2] as i32, rgbs[2][2] as i32, rgbs[3][2] as i32, rgbs[4][2] as i32, rgbs[5][2] as i32, rgbs[6][2] as i32, rgbs[7][2] as i32);
    (r, g, b)
}

unsafe fn rgbs_to_xyzs(r: __m256, g: __m256, b: __m256) -> (__m256, __m256, __m256) {
    // let (r, g, b) = clamp(r, g, b);

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
unsafe fn uint_to_f32(r: __m256i, g: __m256i, b: __m256i) -> (__m256, __m256, __m256) {
    (
        _mm256_cvtepi32_ps(r),
        _mm256_cvtepi32_ps(g),
        _mm256_cvtepi32_ps(b),
    )
}

#[inline]
unsafe fn rgbs_to_xyzs_map(c: __m256) -> __m256 {
    let mask = _mm256_cmp_ps(c, _mm256_set1_ps(10.0), _CMP_GT_OQ);
    let true_branch = {
        const A: f32 = 0.055 * 255.0;
        const D: f32 = 1.055 * 255.0;
        let t0 = _mm256_div_ps(_mm256_add_ps(c, _mm256_set1_ps(A)), _mm256_set1_ps(D));

        let mut unpacked: [f32; 8] = mem::transmute(t0);
        for el in unpacked.iter_mut() {
            *el = el.powf(2.4);
        }
        mem::transmute(unpacked)
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

    let l = _mm256_add_ps(_mm256_mul_ps(y, _mm256_set1_ps(116.0)), _mm256_set1_ps(-16.0));
    let a = _mm256_mul_ps(_mm256_sub_ps(x, y), _mm256_set1_ps(500.0));
    let b = _mm256_mul_ps(_mm256_sub_ps(y, z), _mm256_set1_ps(200.0));

    (l, a, b)
}

#[inline]
unsafe fn xyzs_to_labs_map(c: __m256) -> __m256 {
    // do false branch first
    let false_branch = _mm256_div_ps(
        _mm256_add_ps(
            _mm256_mul_ps(c, _mm256_set1_ps(KAPPA)),
            _mm256_set1_ps(16.0)),
        _mm256_set1_ps(116.0));

    let unpacked_false_branch: [f32; 8] = mem::transmute(false_branch);
    let mut unpacked: [f32; 8] = mem::transmute(c);
    for (i, el) in unpacked.iter_mut().enumerate() {
        if *el > EPSILON {
            *el = el.powf(1.0/3.0);
        } else {
            *el = unpacked_false_branch[i];
        }
    }
    mem::transmute(unpacked)
}

unsafe fn simd_to_lab_vec(l: __m256, a: __m256, b: __m256) -> Vec<Lab> {
    let l: [f32; 8] = mem::transmute(l);
    let a: [f32; 8] = mem::transmute(a);
    let b: [f32; 8] = mem::transmute(b);

    l.iter().zip(a.iter()).zip(b.iter()).map(|((&l, &a), &b)| Lab { l, a, b }).rev().collect()
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

#[cfg(all(test, target_feature = "avx", target_feature = "sse4.1"))]
mod test {
    use rand;
    use rand::Rng;
    use rand::distributions::Standard;
    use super::super::super::{Lab, rgbs_to_labs, labs_to_rgbs, avx};

    lazy_static! {
        static ref RGBS: Vec<[u8;3]> = {
            let rand_seed = [0u8;32];
            let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
            rng.sample_iter(&Standard).take(512).collect()
        };
    }

    #[test]
    fn test_avx_rgbs_to_labs() {
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

        let labs_non_avx = rgbs_to_labs(&rgbs);
        let labs_avx = unsafe { avx::rgbs_to_labs(&rgbs) };
        assert_eq!(labs_avx, labs_non_avx);
    }

    #[test]
    fn test_avx_rgbs_to_labs_many() {
        let labs_non_avx = rgbs_to_labs(&RGBS);
        let labs_avx = unsafe { avx::rgbs_to_labs(&RGBS) };
        assert_eq!(labs_avx, labs_non_avx);
    }

    #[test]
    fn test_avx_rgbs_to_labs_unsaturated() {
        let rgbs = vec![
            [253, 120, 138],
        ];
        let labs_non_avx = rgbs_to_labs(&rgbs);
        let labs_avx = unsafe { avx::rgbs_to_labs(&rgbs) };
        assert_eq!(labs_avx, labs_non_avx);
    }
}
