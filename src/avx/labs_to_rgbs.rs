use std::arch::x86_64::*;
use std::{mem, f32};
use super::Lab;

#[allow(dead_code)]
pub unsafe fn labs_to_rgbs(labs: &[Lab]) -> Vec<[u8; 3]> {
    labs.chunks(8).fold(Vec::with_capacity(labs.len()), |mut v, labs| {
        let rgbs: Vec<_> = match labs {
            packed @ &[_, _, _, _, _, _, _, _] => {
                let (l, a, b) = lab_slice_to_simd(packed);
                let (x, y, z) = labs_to_xyzs(l, a, b);
                let (r, g, b) = xyzs_to_rgbs(x, y, z);
                let (r, g, b) = normalize_unit_to_short(r, g, b);
                simd_to_rgb_vec(r, g, b)
            },
            rest => {
                let mut labs: Vec<Lab> = Vec::with_capacity(8);
                labs.extend_from_slice(&rest);
                let num_padding = 8 - rest.len();
                for _ in 0..num_padding {
                    labs.push(Lab { l: f32::NAN, a: f32::NAN, b: f32::NAN });
                }
                let (l, a, b) = lab_slice_to_simd(&labs);
                let (x, y, z) = labs_to_xyzs(l, a, b);
                let (r, g, b) = xyzs_to_rgbs(x, y, z);
                let (r, g, b) = normalize_unit_to_short(r, g, b);
                let mut labs = simd_to_rgb_vec(r, g, b);
                labs.truncate(rest.len());
                labs
            },
        };
        v.extend_from_slice(&rgbs);
        v
    })
}

unsafe fn lab_slice_to_simd(labs: &[Lab]) -> (__m256, __m256, __m256) {
    assert_eq!(labs.len(), 8);
    let lab0 = labs.get_unchecked(0);
    let lab1 = labs.get_unchecked(1);
    let lab2 = labs.get_unchecked(2);
    let lab3 = labs.get_unchecked(3);
    let lab4 = labs.get_unchecked(4);
    let lab5 = labs.get_unchecked(5);
    let lab6 = labs.get_unchecked(6);
    let lab7 = labs.get_unchecked(7);
    let l = _mm256_set_ps(lab0.l, lab1.l, lab2.l, lab3.l, lab4.l, lab5.l, lab6.l, lab7.l);
    let a = _mm256_set_ps(lab0.a, lab1.a, lab2.a, lab3.a, lab4.a, lab5.a, lab6.a, lab7.a);
    let b = _mm256_set_ps(lab0.b, lab1.b, lab2.b, lab3.b, lab4.b, lab5.b, lab6.b, lab7.b);
    (l, a, b)
}

unsafe fn labs_to_xyzs(l: __m256, a: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let fy = _mm256_div_ps(_mm256_add_ps(l, _mm256_set1_ps(16.0)), _mm256_set1_ps(116.0));
    let fx = _mm256_add_ps(_mm256_div_ps(a, _mm256_set1_ps(500.0)), fy);
    let fz = _mm256_sub_ps(fy, _mm256_div_ps(b, _mm256_set1_ps(200.0)));

    let xr = {
        let false_branch = {
            let temp1 = _mm256_mul_ps(fx, _mm256_set1_ps(116.0));
            let temp2 = _mm256_sub_ps(temp1, _mm256_set1_ps(16.0));
            _mm256_div_ps(temp2, _mm256_set1_ps(903.3))
        };
        let unpacked_false_branch: [f32; 8] = mem::transmute(false_branch);
        let mut unpacked: [f32; 8] = mem::transmute(fx);
        for (i, el) in unpacked.iter_mut().enumerate() {
            let raised = el.powi(3);
            if raised > 0.008856 {
                *el = raised;
            } else {
                *el = unpacked_false_branch[i];
            }
        }
        mem::transmute(unpacked)
    };

    let yr = {
        let false_branch = _mm256_div_ps(l, _mm256_set1_ps(903.3));
        let true_branch = {
            let mut unpacked: [f32; 8] = mem::transmute(fy);
            for el in unpacked.iter_mut() {
                *el = el.powi(3);
            }
            mem::transmute(unpacked)
        };
        let mask = _mm256_cmp_ps(l, _mm256_set1_ps(0.008856 * 903.3), _CMP_GT_OQ);
        _mm256_blendv_ps(false_branch, true_branch, mask)
    };

    let zr = {
        let false_branch = {
            let temp1 = _mm256_mul_ps(fz, _mm256_set1_ps(116.0));
            let temp2 = _mm256_sub_ps(temp1, _mm256_set1_ps(16.0));
            _mm256_div_ps(temp2, _mm256_set1_ps(903.3))
        };
        let unpacked_false_branch: [f32; 8] = mem::transmute(false_branch);
        let mut unpacked: [f32; 8] = mem::transmute(fz);
        for (i, el) in unpacked.iter_mut().enumerate() {
            let raised = el.powi(3);
            if raised > 0.008856 {
                *el = raised;
            } else {
                *el = unpacked_false_branch[i];
            }
        }
        mem::transmute(unpacked)
    };

    let x = _mm256_mul_ps(xr, _mm256_set1_ps(95.047));
    let y = _mm256_mul_ps(yr, _mm256_set1_ps(100.0));
    let z = _mm256_mul_ps(zr, _mm256_set1_ps(108.883));

    (x, y, z)
}

unsafe fn xyzs_to_rgbs(x: __m256, y: __m256, z: __m256) -> (__m256, __m256, __m256) {
    let divisor = _mm256_set1_ps(100.0);
    let x = _mm256_div_ps(x, divisor);
    let y = _mm256_div_ps(y, divisor);
    let z = _mm256_div_ps(z, divisor);

    let r = {
        let prod_x = _mm256_mul_ps(x, _mm256_set1_ps(3.2406));
        let prod_y = _mm256_mul_ps(y, _mm256_set1_ps(-1.5372));
        let prod_z = _mm256_mul_ps(z, _mm256_set1_ps(-0.4986));
        let sum = _mm256_add_ps(_mm256_add_ps(prod_x, prod_y), prod_z);
        xyzs_to_rgbs_map(sum)
    };
    let g = {
        let prod_x = _mm256_mul_ps(x, _mm256_set1_ps(-0.9689));
        let prod_y = _mm256_mul_ps(y, _mm256_set1_ps(1.8758));
        let prod_z = _mm256_mul_ps(z, _mm256_set1_ps(0.0415));
        let sum = _mm256_add_ps(_mm256_add_ps(prod_x, prod_y), prod_z);
        xyzs_to_rgbs_map(sum)
    };
    let b = {
        let prod_x = _mm256_mul_ps(x, _mm256_set1_ps(0.0557));
        let prod_y = _mm256_mul_ps(y, _mm256_set1_ps(-0.2040));
        let prod_z = _mm256_mul_ps(z, _mm256_set1_ps(1.0570));
        let sum = _mm256_add_ps(_mm256_add_ps(prod_x, prod_y), prod_z);
        xyzs_to_rgbs_map(sum)
    };

    (r, g, b)
}

#[inline]
unsafe fn xyzs_to_rgbs_map(c: __m256) ->  __m256 {
    let mask = _mm256_cmp_ps(c, _mm256_set1_ps(0.0031308), _CMP_GT_OQ);
    let true_branch = {
        let mut unpacked: [f32; 8] = mem::transmute(c);
        for el in unpacked.iter_mut() {
            *el = el.powf(1.0 / 2.4);
        }
        let temp1: __m256 = mem::transmute(unpacked);
        let temp2 = _mm256_mul_ps(temp1, _mm256_set1_ps(1.055));
        _mm256_sub_ps(temp2, _mm256_set1_ps(0.055))
    };
    let false_branch = _mm256_mul_ps(c, _mm256_set1_ps(12.92));
    _mm256_blendv_ps(false_branch, true_branch, mask)
}

unsafe fn simd_to_rgb_vec(r: __m256, g: __m256, b: __m256) -> Vec<[u8; 3]> {
    let r: [f32; 8] = mem::transmute(_mm256_round_ps(r, _MM_FROUND_TO_NEAREST_INT));
    let g: [f32; 8] = mem::transmute(_mm256_round_ps(g, _MM_FROUND_TO_NEAREST_INT));
    let b: [f32; 8] = mem::transmute(_mm256_round_ps(b, _MM_FROUND_TO_NEAREST_INT));
    r.iter().zip(g.iter()).zip(b.iter()).map(|((&r, &g), &b)| [r as u8, g as u8, b as u8]).rev().collect()
}

#[inline]
unsafe fn normalize_unit_to_short(r: __m256, g: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let normalizer = _mm256_set1_ps(255.0);
    let r = _mm256_mul_ps(r, normalizer);
    let g = _mm256_mul_ps(g, normalizer);
    let b = _mm256_mul_ps(b, normalizer);
    (r, g, b)
}

#[cfg(all(test, target_feature = "avx", target_feature = "sse4.1"))]
mod test {
    use rand;
    use rand::Rng;
    use rand::distributions::Standard;
    use super::rgbs_to_labs as rgbs_to_labs_avx;
    use super::labs_to_rgbs as labs_to_rgbs_avx;
    use super::super::{Lab, rgbs_to_labs, labs_to_rgbs};

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
        let labs_avx = unsafe { rgbs_to_labs_avx(&rgbs) };
        assert_eq!(labs_avx, labs_non_avx);
    }

    #[test]
    fn test_avx_rgbs_to_labs_many() {
        let labs_non_avx = rgbs_to_labs(&RGBS);
        let labs_avx = unsafe { rgbs_to_labs_avx(&RGBS) };
        assert_eq!(labs_avx, labs_non_avx);
    }

    #[test]
    fn test_avx_rgbs_to_labs_unsaturated() {
        let rgbs = vec![
            [253, 120, 138],
        ];
        let labs_non_avx = rgbs_to_labs(&rgbs);
        let labs_avx = unsafe { rgbs_to_labs_avx(&rgbs) };
        assert_eq!(labs_avx, labs_non_avx);
    }

    #[test]
    fn test_avx_labs_to_rgbs() {
        let labs = unsafe { rgbs_to_labs_avx(&RGBS) };
        let rgbs = unsafe { labs_to_rgbs_avx(&labs) };
        assert_eq!(rgbs.as_slice(), RGBS.as_slice());
    }

    #[test]
    fn test_avx_labs_to_rgbs_unsaturated() {
        let labs = vec![
            Lab { l: 66.6348, a: 52.260696, b: 14.850557 },
        ];
        let rgbs_non_avx = labs_to_rgbs(&labs);
        let rgbs_avx = unsafe { labs_to_rgbs_avx(&labs) };
        assert_eq!(rgbs_avx, rgbs_non_avx);
    }
}
