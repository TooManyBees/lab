use std::arch::x86_64::*;

use std::{mem, f32};
use super::Lab;

type TuplesF32 = (f32, f32, f32, f32, f32, f32, f32, f32);

unsafe fn avx_load_rgbs(rgbs: &[[f32; 3]]) -> (__m256, __m256, __m256) {
    assert_eq!(rgbs.len(), 8);
    let rgb0 = rgbs.get_unchecked(0);
    let rgb1 = rgbs.get_unchecked(1);
    let rgb2 = rgbs.get_unchecked(2);
    let rgb3 = rgbs.get_unchecked(3);
    let rgb4 = rgbs.get_unchecked(4);
    let rgb5 = rgbs.get_unchecked(5);
    let rgb6 = rgbs.get_unchecked(6);
    let rgb7 = rgbs.get_unchecked(7);
    let r = _mm256_set_ps(rgb0[0], rgb1[0], rgb2[0], rgb3[0], rgb4[0], rgb5[0], rgb6[0], rgb7[0]);
    let g = _mm256_set_ps(rgb0[1], rgb1[1], rgb2[1], rgb3[1], rgb4[1], rgb5[1], rgb6[1], rgb7[1]);
    let b = _mm256_set_ps(rgb0[2], rgb1[2], rgb2[2], rgb3[2], rgb4[2], rgb5[2], rgb6[2], rgb7[2]);
    (r, g, b)
}

unsafe fn avx_dump_labs(l: __m256, a: __m256, b: __m256) -> Vec<Lab> {
    let l: [f32; 8] = mem::transmute(l);
    let a: [f32; 8] = mem::transmute(a);
    let b: [f32; 8] = mem::transmute(b);

    l.iter().zip(a.iter()).zip(b.iter()).map(|((&l, &a), &b)| Lab { l, a, b }).rev().collect()
}

unsafe fn rgbs_to_xyzs_map(c: __m256) -> __m256 {
    let mask = _mm256_cmp_ps(c, _mm256_set1_ps(0.04045), _CMP_GT_OQ);
    let true_branch = {
        let a = _mm256_add_ps(c, _mm256_set1_ps(0.055));
        let b = _mm256_div_ps(a, _mm256_set1_ps(1.055));
        let unpacked: TuplesF32 = mem::transmute(b);
        _mm256_set_ps(
            unpacked.7.powf(2.4),
            unpacked.6.powf(2.4),
            unpacked.5.powf(2.4),
            unpacked.4.powf(2.4),
            unpacked.3.powf(2.4),
            unpacked.2.powf(2.4),
            unpacked.1.powf(2.4),
            unpacked.0.powf(2.4),
        )
    };
    let false_branch = _mm256_div_ps(c, _mm256_set1_ps(12.92));
    // TODO: is it more efficient to mask the branch operations
    // than to blend once?
    let blended = _mm256_blendv_ps(false_branch, true_branch, mask);

    _mm256_mul_ps(blended, _mm256_set1_ps(100.0))
}

unsafe fn rgbs_to_xyzs(r: __m256, g: __m256, b: __m256) -> (__m256, __m256, __m256) {
    let normalizer = _mm256_set1_ps(255.0);
    let r = _mm256_div_ps(r, normalizer);
    let g = _mm256_div_ps(g, normalizer);
    let b = _mm256_div_ps(b, normalizer);

    let r = rgbs_to_xyzs_map(r);
    let g = rgbs_to_xyzs_map(g);
    let b = rgbs_to_xyzs_map(b);

    let x = {
        let prod_r = _mm256_mul_ps(r, _mm256_set1_ps(0.4124));
        let prod_g = _mm256_mul_ps(g, _mm256_set1_ps(0.3576));
        let prod_b = _mm256_mul_ps(b, _mm256_set1_ps(0.1805));
        _mm256_add_ps(_mm256_add_ps(prod_r, prod_g), prod_b)
    };

    let y = {
        let prod_r = _mm256_mul_ps(r, _mm256_set1_ps(0.2126));
        let prod_g = _mm256_mul_ps(g, _mm256_set1_ps(0.7152));
        let prod_b = _mm256_mul_ps(b, _mm256_set1_ps(0.0722));
        _mm256_add_ps(_mm256_add_ps(prod_r, prod_g), prod_b)
    };

    let z = {
        let prod_r = _mm256_mul_ps(r, _mm256_set1_ps(0.0193));
        let prod_g = _mm256_mul_ps(g, _mm256_set1_ps(0.1192));
        let prod_b = _mm256_mul_ps(b, _mm256_set1_ps(0.9505));
        _mm256_add_ps(_mm256_add_ps(prod_r, prod_g), prod_b)
    };

    (x, y, z)
}

unsafe fn xyzs_to_labs_map(c: __m256) -> __m256 {
    // do false branch first
    let false_branch = _mm256_mul_ps(c, _mm256_set1_ps(7.787));
    let false_branch = _mm256_add_ps(_mm256_set1_ps(16.0 / 116.0), false_branch);

    let unpacked_false_branch: [f32; 8] = mem::transmute(false_branch);
    let mut unpacked: [f32; 8] = mem::transmute(c);
    for (i, el) in unpacked.iter_mut().enumerate() {
        if *el > 0.008856 {
            *el = el.powf(1.0/3.0);
        } else {
            *el = unpacked_false_branch[i];
        }
    }
    mem::transmute(unpacked)
}

unsafe fn xyzs_to_labs(x: __m256, y: __m256, z: __m256) -> (__m256, __m256, __m256) {
    let x = xyzs_to_labs_map(_mm256_div_ps(x, _mm256_set1_ps(95.047)));
    let y = xyzs_to_labs_map(_mm256_div_ps(y, _mm256_set1_ps(100.0)));
    let z = xyzs_to_labs_map(_mm256_div_ps(z, _mm256_set1_ps(108.883)));

    let l = _mm256_add_ps(_mm256_mul_ps(y, _mm256_set1_ps(116.0)), _mm256_set1_ps(-16.0));
    let a = _mm256_mul_ps(_mm256_sub_ps(x, y), _mm256_set1_ps(500.0));
    let b = _mm256_mul_ps(_mm256_sub_ps(y, z), _mm256_set1_ps(200.0));

    (l, a, b)
}

#[allow(dead_code)]
pub unsafe fn rgbs_to_labs(rgbs: &[[u8; 3]]) -> Vec<Lab> {
    rgbs.chunks(8).fold(Vec::with_capacity(rgbs.len()), |mut v, rgbs| {
        let labs = match rgbs {
            &[rgb0, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7] => {
                let (r, g, b) = avx_load_rgbs(&[
                    [rgb0[0] as f32, rgb0[1] as f32, rgb0[2] as f32],
                    [rgb1[0] as f32, rgb1[1] as f32, rgb1[2] as f32],
                    [rgb2[0] as f32, rgb2[1] as f32, rgb2[2] as f32],
                    [rgb3[0] as f32, rgb3[1] as f32, rgb3[2] as f32],
                    [rgb4[0] as f32, rgb4[1] as f32, rgb4[2] as f32],
                    [rgb5[0] as f32, rgb5[1] as f32, rgb5[2] as f32],
                    [rgb6[0] as f32, rgb6[1] as f32, rgb6[2] as f32],
                    [rgb7[0] as f32, rgb7[1] as f32, rgb7[2] as f32],
                ]);
                let (x, y, z) = rgbs_to_xyzs(r, g, b);
                let (l, a, b) = xyzs_to_labs(x, y, z);
                avx_dump_labs(l, a, b)
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
                let (r, g, b) = avx_load_rgbs(rgbs.as_slice());
                let (x, y, z) = rgbs_to_xyzs(r, g, b);
                let (l, a, b) = xyzs_to_labs(x, y, z);
                let mut labs = avx_dump_labs(l, a, b);
                labs.truncate(rest.len());
                labs
            },
        };
        v.extend_from_slice(&labs);
        v
    })
}
