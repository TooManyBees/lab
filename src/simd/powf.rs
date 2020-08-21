use std::arch::x86_64::*;

unsafe fn scalbnf(mut x: __m256, mut n: __m256i) -> __m256 {
    let x1p127 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f000000)); // 0x1p127f === 2 ^ 127
    let x1p_126 = _mm256_castsi256_ps(_mm256_set1_epi32(0x800000)); // 0x1p-126f === 2 ^ -126
    let x1p24 = _mm256_castsi256_ps(_mm256_set1_epi32(0x4b800000)); // 0x1p24f === 2 ^ 24
    let x1p_126_x1p24 = _mm256_mul_ps(x1p_126, x1p24);

    let mut mask1 = _mm256_cmpgt_epi32(n, _mm256_set1_epi32(127));
    let mask1_copy = mask1.clone();
    x = _mm256_blendv_ps(x, _mm256_mul_ps(x, x1p127), _mm256_castsi256_ps(mask1));
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_sub_epi32(n, _mm256_set1_epi32(127))),
        _mm256_castsi256_ps(mask1),
    ));
    mask1 = _mm256_and_si256(mask1, _mm256_cmpgt_epi32(n, _mm256_set1_epi32(127)));
    x = _mm256_blendv_ps(x, _mm256_mul_ps(x, x1p127), _mm256_castsi256_ps(mask1));
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_sub_epi32(n, _mm256_set1_epi32(127))),
        _mm256_castsi256_ps(mask1),
    ));
    mask1 = _mm256_and_si256(mask1, _mm256_cmpgt_epi32(n, _mm256_set1_epi32(127)));
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_set1_epi32(127)),
        _mm256_castsi256_ps(mask1),
    ));

    let mut mask2 = _mm256_andnot_si256(mask1_copy, _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), n));
    x = _mm256_blendv_ps(
        x,
        _mm256_mul_ps(x, x1p_126_x1p24),
        _mm256_castsi256_ps(mask2),
    );
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_add_epi32(n, _mm256_set1_epi32(126 - 24))),
        _mm256_castsi256_ps(mask2),
    ));
    mask2 = _mm256_and_si256(mask2, _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), n));
    x = _mm256_blendv_ps(
        x,
        _mm256_mul_ps(x, x1p_126_x1p24),
        _mm256_castsi256_ps(mask2),
    );
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_add_epi32(n, _mm256_set1_epi32(126 - 24))),
        _mm256_castsi256_ps(mask2),
    ));
    mask2 = _mm256_and_si256(mask2, _mm256_cmpgt_epi32(_mm256_set1_epi32(-126), n));
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_set1_epi32(-126)),
        _mm256_castsi256_ps(mask2),
    ));

    _mm256_mul_ps(
        x,
        _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(_mm256_set1_epi32(0x7f), n),
            23,
        )),
    )
}

unsafe fn fabsf(x: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_and_si256(
        _mm256_castps_si256(x),
        _mm256_set1_epi32(0x7fffffff),
    ))
}

/* origin: FreeBSD /usr/src/lib/msun/src/e_powf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

const BP: [f32; 2] = [1.0, 1.5];
const BP_0: f32 = 1.0;
const BP_1: f32 = 1.5;
const DP_H: [f32; 2] = [0.0, 5.84960938e-01]; /* 0x3f15c000 */
const DP_H_0: f32 = 0.0;
const DP_H_1: f32 = 5.84960938e-01;
const DP_L: [f32; 2] = [0.0, 1.56322085e-06]; /* 0x35d1cfdc */
const DP_L_0: f32 = 0.0;
const DP_L_1: f32 = 1.56322085e-06;
const TWO24: f32 = 16777216.0; /* 0x4b800000 */
const HUGE: f32 = 1.0e30;
const TINY: f32 = 1.0e-30;
const L1: f32 = 6.0000002384e-01; /* 0x3f19999a */
const L2: f32 = 4.2857143283e-01; /* 0x3edb6db7 */
const L3: f32 = 3.3333334327e-01; /* 0x3eaaaaab */
const L4: f32 = 2.7272811532e-01; /* 0x3e8ba305 */
const L5: f32 = 2.3066075146e-01; /* 0x3e6c3255 */
const L6: f32 = 2.0697501302e-01; /* 0x3e53f142 */
const P1: f32 = 1.6666667163e-01; /* 0x3e2aaaab */
const P2: f32 = -2.7777778450e-03; /* 0xbb360b61 */
const P3: f32 = 6.6137559770e-05; /* 0x388ab355 */
const P4: f32 = -1.6533901999e-06; /* 0xb5ddea0e */
const P5: f32 = 4.1381369442e-08; /* 0x3331bb4c */
const LG2: f32 = 6.9314718246e-01; /* 0x3f317218 */
const LG2_H: f32 = 6.93145752e-01; /* 0x3f317200 */
const LG2_L: f32 = 1.42860654e-06; /* 0x35bfbe8c */
const OVT: f32 = 4.2995665694e-08; /* -(128-log2(ovfl+.5ulp)) */
const CP: f32 = 9.6179670095e-01; /* 0x3f76384f =2/(3ln2) */
const CP_H: f32 = 9.6191406250e-01; /* 0x3f764000 =12b cp */
const CP_L: f32 = -1.1736857402e-04; /* 0xb8f623c6 =tail of cp_h */
const IVLN2: f32 = 1.4426950216e+00;
const IVLN2_H: f32 = 1.4426879883e+00;
const IVLN2_L: f32 = 7.0526075433e-06;

pub unsafe fn powf(x: __m256, y: __m256) -> __m256 {
    let mut z: __m256;
    let mut ax: __m256;
    let z_h: __m256;
    let z_l: __m256;
    let mut p_h: __m256;
    let mut p_l: __m256;
    let y1: __m256;
    let mut t1: __m256;
    let t2: __m256;
    let mut r: __m256;
    let s: __m256;
    let mut sn: __m256;
    let mut t: __m256;
    let mut u: __m256;
    let mut v: __m256;
    let mut w: __m256;
    let i: __m256i;
    let mut j: __m256i;
    let mut k: __m256i;
    let mut yisint: __m256i;
    let mut n: __m256i;
    let hx: __m256i;
    let hy: __m256i;
    let mut ix: __m256i;
    let iy: __m256i;
    let mut is: __m256i;

    hx = _mm256_castps_si256(x);
    hy = _mm256_castps_si256(y);

    ix = _mm256_and_si256(hx, _mm256_set1_epi32(0x7fffffff));
    iy = _mm256_and_si256(hy, _mm256_set1_epi32(0x7fffffff));

    // if y is +-inf
    //   if |x| == 1, return 1
    //   if |x| > 1
    //     if y >= 0, return inf
    //     if y < 0, return 0
    //   if |x| < 1
    //     if y >= 0, return 0
    //     if y < 0, return -inf

    // if y == -1, return 1 / x

    ax = fabsf(x);

    let inf_mask = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_cmpeq_epi32(ix, _mm256_set1_epi32(0x7f800000)),
            _mm256_cmpeq_epi32(ix, _mm256_set1_epi32(0x3f800000)),
        ),
        _mm256_cmpeq_epi32(ix, _mm256_set1_epi32(0)),
    );
    let zero_mask = _mm256_cmpeq_epi32(ix, _mm256_set1_epi32(0));

    sn = _mm256_set1_ps(1.0); /* sign of result */

    // x < 0 ** nonInt, return NaN
    // x < 0 and odd int y, flip sn

    let mut s2: __m256;
    let mut s_h: __m256;
    let s_l: __m256;
    let mut t_h: __m256;
    let mut t_l: __m256;

    n = _mm256_set1_epi32(0);

    /* take care subnormal number */
    let subnormal_mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), ix);
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(_mm256_set1_epi32(0)),
        _mm256_castsi256_ps(_mm256_set1_epi32(-24)),
        _mm256_castsi256_ps(subnormal_mask),
    ));
    ax = _mm256_blendv_ps(
        ax,
        _mm256_mul_ps(ax, _mm256_set1_ps(TWO24)),
        _mm256_castsi256_ps(subnormal_mask),
    );
    ix = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(ix),
        ax,
        _mm256_castsi256_ps(subnormal_mask),
    ));

    n = _mm256_add_epi32(
        n,
        _mm256_sub_epi32(_mm256_srli_epi32(ix, 23), _mm256_set1_epi32(0x7f)),
    );

    j = _mm256_and_si256(ix, _mm256_set1_epi32(0x007fffff));

    /* determine interval */
    ix = _mm256_or_si256(j, _mm256_set1_epi32(0x3f800000)); /* normalize ix */
    {
        let mask_k = _mm256_and_si256(
            _mm256_cmpgt_epi32(j, _mm256_set1_epi32(0x1cc471)),
            _mm256_cmpgt_epi32(_mm256_set1_epi32(0x5db3d7), j),
        );
        let mask_n = _mm256_or_si256(
            _mm256_cmpeq_epi32(j, _mm256_set1_epi32(0x5db3d7)),
            _mm256_cmpgt_epi32(j, _mm256_set1_epi32(0x5db3d7)),
        );

        k = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(_mm256_set1_epi32(0)),
            _mm256_castsi256_ps(_mm256_set1_epi32(1)),
            _mm256_castsi256_ps(mask_k),
        ));
        n = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(n),
            _mm256_castsi256_ps(_mm256_add_epi32(n, _mm256_set1_epi32(1))),
            _mm256_castsi256_ps(mask_n),
        ));
        ix = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(ix),
            _mm256_castsi256_ps(_mm256_sub_epi32(ix, _mm256_set1_epi32(0x00800000))),
            _mm256_castsi256_ps(mask_n),
        ));
    }

    ax = _mm256_castsi256_ps(ix);

    // let bp = _mm256_i32gather_ps(&BP as *const f32, k, 4);
    let bp = _mm256_blendv_ps(
        _mm256_set1_ps(BP_0),
        _mm256_set1_ps(BP_1),
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_set1_epi32(1), k)),
    );
    u = _mm256_sub_ps(ax, bp); /* bp[0]=1.0, bp[1]=1.5 */
    v = _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_add_ps(ax, bp));
    s = _mm256_mul_ps(u, v);
    s_h = s;
    is = _mm256_castps_si256(s_h);
    s_h = _mm256_castsi256_ps(_mm256_and_si256(
        is,
        _mm256_set1_epi32(-4096i32 /* 0xFFFFF000 */),
    ));

    /* t_h=ax+bp[k] High */
    is = _mm256_or_si256(
        _mm256_and_si256(
            _mm256_srli_epi32(ix, 1),
            _mm256_set1_epi32(-4096i32 /* 0xFFFFF000 */),
        ),
        _mm256_set1_epi32(0x20000000),
    );
    t_h = _mm256_castsi256_ps(_mm256_add_epi32(
        _mm256_add_epi32(is, _mm256_set1_epi32(0x00400000)),
        _mm256_slli_epi32(k, 21),
    ));
    t_l = _mm256_sub_ps(ax, _mm256_sub_ps(t_h, bp));
    s_l = _mm256_mul_ps(
        v,
        _mm256_sub_ps(
            _mm256_sub_ps(u, _mm256_mul_ps(s_h, t_h)),
            _mm256_mul_ps(s_h, t_l),
        ),
    );

    /* compute log(ax) */
    s2 = _mm256_mul_ps(s, s);
    r = _mm256_mul_ps(
        _mm256_mul_ps(s2, s2),
        _mm256_add_ps(
            _mm256_set1_ps(L1),
            _mm256_mul_ps(
                s2,
                _mm256_add_ps(
                    _mm256_set1_ps(L2),
                    _mm256_mul_ps(
                        s2,
                        _mm256_add_ps(
                            _mm256_set1_ps(L3),
                            _mm256_mul_ps(
                                s2,
                                _mm256_add_ps(
                                    _mm256_set1_ps(L4),
                                    _mm256_mul_ps(
                                        s2,
                                        _mm256_add_ps(
                                            _mm256_set1_ps(L5),
                                            _mm256_mul_ps(s2, _mm256_set1_ps(L6)),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );
    r = _mm256_add_ps(r, _mm256_mul_ps(s_l, _mm256_add_ps(s_h, s)));
    s2 = _mm256_mul_ps(s_h, s_h);
    t_h = _mm256_add_ps(_mm256_set1_ps(3.0), _mm256_add_ps(s2, r));
    is = _mm256_castps_si256(t_h);
    t_h = _mm256_castsi256_ps(_mm256_and_si256(
        is,
        _mm256_set1_epi32(-4096i32 /* 0xFFFFF000 */),
    ));
    t_l = _mm256_sub_ps(
        r,
        _mm256_sub_ps(_mm256_sub_ps(t_h, _mm256_set1_ps(3.0)), s2),
    );

    /* u+v = s*(1+...) */
    u = _mm256_mul_ps(s_h, t_h);
    v = _mm256_add_ps(_mm256_mul_ps(s_l, t_h), _mm256_mul_ps(t_l, s));

    /* 2/(3log2)*(s+...) */
    p_h = _mm256_add_ps(u, v);
    is = _mm256_castps_si256(p_h);
    p_h = _mm256_castsi256_ps(_mm256_and_si256(
        is,
        _mm256_set1_epi32(-4096i32 /* 0xFFFFF000 */),
    ));
    p_l = _mm256_sub_ps(v, _mm256_sub_ps(p_h, u));
    z_h = _mm256_mul_ps(_mm256_set1_ps(CP_H), p_h); /* cp_h+cp_l = 2/(3*log2) */
    z_l = _mm256_add_ps(
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(CP_L), p_h),
            _mm256_mul_ps(p_l, _mm256_set1_ps(CP)),
        ),
        // _mm256_i32gather_ps(DP_L.as_ptr(), k, 4),
        _mm256_blendv_ps(
            _mm256_set1_ps(DP_L_0),
            _mm256_set1_ps(DP_L_1),
            _mm256_castsi256_ps(_mm256_cmpeq_epi32(k, _mm256_set1_epi32(1))),
        ),
    );

    /* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
    t = _mm256_cvtepi32_ps(n);
    t1 = _mm256_add_ps(
        _mm256_add_ps(
            _mm256_add_ps(z_h, z_l),
            // _mm256_i32gather_ps(DP_H.as_ptr(), k, 4),
            _mm256_blendv_ps(
                _mm256_set1_ps(DP_H_0),
                _mm256_set1_ps(DP_H_1),
                _mm256_castsi256_ps(_mm256_cmpeq_epi32(k, _mm256_set1_epi32(1))),
            ),
        ),
        t,
    );
    is = _mm256_castps_si256(t1);
    t1 = _mm256_castsi256_ps(_mm256_and_si256(
        is,
        _mm256_set1_epi32(-4096i32 /* 0xFFFFF000 */),
    ));
    t2 = _mm256_sub_ps(
        z_l,
        _mm256_sub_ps(
            _mm256_sub_ps(
                _mm256_sub_ps(t1, t),
                // _mm256_i32gather_ps(DP_H.as_ptr(), k, 4),
                _mm256_blendv_ps(
                    _mm256_set1_ps(DP_H_0),
                    _mm256_set1_ps(DP_H_1),
                    _mm256_castsi256_ps(_mm256_cmpeq_epi32(k, _mm256_set1_epi32(1))),
                ),
            ),
            z_h,
        ),
    );

    /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
    is = _mm256_castps_si256(y);
    y1 = _mm256_castsi256_ps(_mm256_and_si256(
        is,
        _mm256_set1_epi32(-4096i32 /* 0xFFFFF000 */),
    ));
    p_l = _mm256_add_ps(
        _mm256_mul_ps(_mm256_sub_ps(y, y1), t1),
        _mm256_mul_ps(y, t2),
    );
    p_h = _mm256_mul_ps(y1, t1);
    z = _mm256_add_ps(p_l, p_h);
    j = _mm256_castps_si256(z);
    // TODO: check for various overflow, underflow, etc

    /*
     * compute 2**(p_h+p_l)
     */
    i = _mm256_and_si256(j, _mm256_set1_epi32(0x7fffffff));
    k = _mm256_sub_epi32(_mm256_srli_epi32(i, 23), _mm256_set1_epi32(0x7f));
    n = _mm256_set1_epi32(0);

    let mask = _mm256_cmpgt_epi32(i, _mm256_set1_epi32(0x3f000000));
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_add_epi32(
            j,
            _mm256_srlv_epi32(
                _mm256_set1_epi32(0x00800000),
                _mm256_add_epi32(k, _mm256_set1_epi32(1)),
            ),
        )),
        _mm256_castsi256_ps(mask),
    ));
    k = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(k),
        _mm256_castsi256_ps(_mm256_sub_epi32(
            _mm256_srli_epi32(_mm256_and_si256(n, _mm256_set1_epi32(0x7fffffff)), 23),
            _mm256_set1_epi32(0x7f),
        )),
        _mm256_castsi256_ps(mask),
    ));
    t = _mm256_blendv_ps(
        t,
        _mm256_castsi256_ps(_mm256_andnot_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(0x007fffff), k),
            n,
        )),
        _mm256_castsi256_ps(mask),
    );
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_srlv_epi32(
            _mm256_or_si256(
                _mm256_and_si256(n, _mm256_set1_epi32(0x007fffff)),
                _mm256_set1_epi32(0x00800000),
            ),
            _mm256_sub_epi32(_mm256_set1_epi32(23), k),
        )),
        _mm256_castsi256_ps(mask),
    ));
    let mask2 = _mm256_and_si256(mask, _mm256_cmpgt_epi32(_mm256_set1_epi32(0), j));
    n = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(n),
        _mm256_castsi256_ps(_mm256_sub_epi32(_mm256_set1_epi32(0), n)),
        _mm256_castsi256_ps(mask2),
    ));
    p_h = _mm256_blendv_ps(p_h, _mm256_sub_ps(p_h, t), _mm256_castsi256_ps(mask));
    t = _mm256_add_ps(p_l, p_h);
    is = _mm256_castps_si256(t);
    t = _mm256_castsi256_ps(_mm256_and_si256(
        is,
        _mm256_set1_epi32(-32768i32 /* 0xffff8000 */),
    ));
    u = _mm256_mul_ps(t, _mm256_set1_ps(LG2_H));
    v = _mm256_add_ps(
        _mm256_mul_ps(
            _mm256_sub_ps(p_l, _mm256_sub_ps(t, p_h)),
            _mm256_set1_ps(LG2),
        ),
        _mm256_mul_ps(t, _mm256_set1_ps(LG2_L)),
    );
    z = _mm256_add_ps(u, v);
    w = _mm256_sub_ps(v, _mm256_sub_ps(z, u));
    t = _mm256_mul_ps(z, z);
    t1 = _mm256_sub_ps(
        z,
        _mm256_mul_ps(
            t,
            _mm256_add_ps(
                _mm256_set1_ps(P1),
                _mm256_mul_ps(
                    t,
                    _mm256_add_ps(
                        _mm256_set1_ps(P2),
                        _mm256_mul_ps(
                            t,
                            _mm256_add_ps(
                                _mm256_set1_ps(P3),
                                _mm256_mul_ps(
                                    t,
                                    _mm256_add_ps(
                                        _mm256_set1_ps(P4),
                                        _mm256_mul_ps(t, _mm256_set1_ps(P5)),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );
    r = _mm256_sub_ps(
        _mm256_div_ps(_mm256_mul_ps(z, t1), _mm256_sub_ps(t1, _mm256_set1_ps(2.0))),
        _mm256_add_ps(w, _mm256_mul_ps(z, w)),
    );
    z = _mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_sub_ps(r, z));
    j = _mm256_castps_si256(z);
    j = _mm256_add_epi32(j, _mm256_slli_epi32(n, 23));
    z = _mm256_blendv_ps(
        scalbnf(z, n),
        _mm256_castsi256_ps(j),
        _mm256_castsi256_ps(_mm256_cmpgt_epi32(
            _mm256_srli_epi32(j, 23),
            _mm256_set1_epi32(0),
        )),
    );

    let mut ret = _mm256_mul_ps(sn, z);

    /* special value of x */
    ret = _mm256_blendv_ps(
        ret,
        x,
        _mm256_castsi256_ps(zero_mask),
    );

    // /* special value of y */
    // let y_lt_zero_mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(0), hy));
    // /* y is  0.5 */
    // // ret = _mm256_blendv_ps(
    // //     ret,
    // //     sqrt(x),
    // //     _mm256_castsi256_ps(_mm256_and_si256(_mm256_cmpeq_epi32(hy, _mm256_set1_epi32(0x3f000000)), _mm256_cmpgt_epi32(_mm256_set1_epi32(0), hx))),
    // // );
    // /* y is 2 */
    // ret = _mm256_blendv_ps(
    //     ret,
    //     _mm256_mul_ps(x, x),
    //     _mm256_castsi256_ps(_mm256_cmpeq_epi32(hy, _mm256_set1_epi32(0x40000000))),
    // );
    // /* y is +-1 */
    // ret = _mm256_blendv_ps(
    //     ret,
    //     _mm256_blendv_ps(
    //         x,
    //         _mm256_div_ps(_mm256_set1_ps(1.0), x),
    //         y_lt_zero_mask,
    //     ),
    //     _mm256_castsi256_ps(_mm256_cmpeq_epi32(iy, _mm256_set1_epi32(0x3f800000))),
    // );
    // let special_value_mask = _mm256_cmpeq_epi32(iy, _mm256_set1_epi32(0x7f800000));
    // /* (|x|<1)**+-inf = 0,inf */
    // ret = _mm256_blendv_ps(
    //     ret,
    //     _mm256_blendv_ps(
    //         _mm256_set1_ps(0.0),
    //         _mm256_mul_ps(y, _mm256_set1_ps(-1.0)),
    //         y_lt_zero_mask,
    //     ),
    //     _mm256_castsi256_ps(special_value_mask)
    // );
    // /* (|x|>1)**+-inf = inf,0 */
    // ret = _mm256_blendv_ps(
    //     ret,
    //     _mm256_blendv_ps(
    //         y,
    //         _mm256_set1_ps(0.0),
    //         y_lt_zero_mask,
    //     ),
    //     _mm256_castsi256_ps(_mm256_and_si256(special_value_mask, _mm256_cmpgt_epi32(ix, _mm256_set1_epi32(0x3f800000)))),
    // );
    // /* (-1)**+-inf is 1 */
    // ret = _mm256_blendv_ps(
    //     ret,
    //     _mm256_set1_ps(1.0),
    //     _mm256_castsi256_ps(_mm256_and_si256(special_value_mask, _mm256_cmpeq_epi32(ix, _mm256_set1_epi32(0x3f800000)))),
    // );

    // /* NaN if either arg is NaN */
    // ret = _mm256_blendv_ps(
    //     ret,
    //     _mm256_add_ps(x, y),
    //     _mm256_castsi256_ps(_mm256_or_si256(
    //         _mm256_cmpgt_epi32(ix, _mm256_set1_epi32(0x7f800000)),
    //         _mm256_cmpgt_epi32(iy, _mm256_set1_epi32(0x7f800000)),
    //     )),
    // );
    // /* x**0 = 1, even if x is NaN */
    // /* 1**y = 1, even if y is NaN */
    // ret = _mm256_blendv_ps(
    //     ret,
    //     _mm256_set1_ps(1.0),
    //     _mm256_castsi256_ps(_mm256_or_si256(
    //         _mm256_cmpeq_epi32(iy, _mm256_set1_epi32(0)),
    //         _mm256_cmpeq_epi32(hx, _mm256_set1_epi32(0x3f800000)),
    //     )),
    // );

    ret
}
