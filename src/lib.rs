//! # Lab
//!
//! Tools for converting RGB colors to L\*a\*b\* measurements.
//!
//! RGB colors, for this crate at least, are considered to be an array
//! of `u8` values (`[u8; 3]`), while L\*a\*b\* colors are represented
//! by its own struct that uses `f32` values.
//!
//! # Usage
//! ## Converting single values
//! To convert a single value, use one of the functions
//! * `lab::Lab::from_rgb(rgb: &[u8; 3]) -> Lab`
//! * `lab::Lab::from_rgba(rgba: &[u8; 4]) -> Lab` (drops the fourth alpha byte)
//! * `lab::Lab::to_rgb(&self) -> [u8; 3]`
//!
//! ## Converting multiple values
//! To convert slices of values
//! * `lab::rgbs_to_labs(rgbs: &[[u8; 3]]) -> Vec<Lab>`
//! * `lab::labs_to_rgbs(labs: &[Lab]) -> Vec<[u8; 3]>`
//! To convert slices using SIMD (AVX, SSE 4.1) operations
//! * `lab::simd::rgbs_to_labs`
//! * `lab::simd::labs_to_rgbs`
//!
//! ## Parallelization concerns
//! This crate makes no assumptions about how to parallelize work, so the above
//! functions that convert slices do so in serial. Presently, parallelizing the
//! functions that accept slices is a manual job of reimplementing
//! them using their fundamental work function, and replacing one iterator method
//! with its equivalent from Rayon.
//!
//! `lab::rgbs_to_labs` and `lab::labs_to_rgbs` are convenience functions for
//! `rgbs.iter().map(Lab::from_rgb).collect()`, which can easily be parallelized
//! with Rayon by replacing `iter()` with `par_iter()`.
//!
//! For the SIMD based functions, their smallest unit of work is done by the
//! functions `lab::simd::rgbs_to_labs_chunk` and `lab::simd::labs_to_rgbs_chunk`
//! which both accept exactly 8 elements. See their respective docs for examples
//! on where to add Rayon methods.

#![doc(html_root_url = "https://docs.rs/lab/0.7.2")]

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;
#[cfg(test)]
#[macro_use]
extern crate lazy_static;
#[cfg(test)]
extern crate rand;

#[cfg(target_arch = "x86_64")]
pub mod simd;

/// Struct representing a color in L\*a\*b\* space
#[derive(Debug, PartialEq, Copy, Clone, Default)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

// κ and ε parameters used in conversion between XYZ and La*b*.  See
// http://www.brucelindbloom.com/LContinuity.html for explanation as to why
// those are different values than those provided by CIE standard.
pub(crate) const KAPPA: f32 = 24389.0 / 27.0;
pub(crate) const EPSILON: f32 = 216.0 / 24389.0;
pub(crate) const CBRT_EPSILON: f32 = 0.20689655172413796;

fn rgb_to_xyz(rgb: &[u8; 3]) -> [f32; 3] {
    let r = rgb_to_xyz_map(rgb[0]);
    let g = rgb_to_xyz_map(rgb[1]);
    let b = rgb_to_xyz_map(rgb[2]);

    [
        r * 0.4124564390896921 + g * 0.357576077643909 + b * 0.18043748326639894,
        r * 0.21267285140562248 + g * 0.715152155287818 + b * 0.07217499330655958,
        r * 0.019333895582329317 + g * 0.119192025881303 + b * 0.9503040785363677,
    ]
}

#[inline]
fn rgb_to_xyz_map(c: u8) -> f32 {
    if c > 10 {
        const A: f32 = 0.055 * 255.0;
        const D: f32 = 1.055 * 255.0;
        ((c as f32 + A) / D).powf(2.4)
    } else {
        const D: f32 = 12.92 * 255.0;
        c as f32 / D
    }
}

fn xyz_to_lab(xyz: [f32; 3]) -> Lab {
    let x = xyz_to_lab_map(xyz[0] / 0.95047);
    let y = xyz_to_lab_map(xyz[1]);
    let z = xyz_to_lab_map(xyz[2] / 1.08883);

    Lab {
        l: (116.0 * y) - 16.0,
        a: 500.0 * (x - y),
        b: 200.0 * (y - z),
    }
}

#[inline]
fn xyz_to_lab_map(c: f32) -> f32 {
    if c > EPSILON {
        c.powf(1.0 / 3.0)
    } else {
        (KAPPA * c + 16.0) / 116.0
    }
}

fn lab_to_xyz(lab: &Lab) -> [f32; 3] {
    let fy = (lab.l + 16.0) / 116.0;
    let fx = (lab.a / 500.0) + fy;
    let fz = fy - (lab.b / 200.0);
    let xr = if fx > CBRT_EPSILON {
        fx.powi(3)
    } else {
        ((fx * 116.0) - 16.0) / KAPPA
    };
    let yr = if lab.l > EPSILON * KAPPA {
        fy.powi(3)
    } else {
        lab.l / KAPPA
    };
    let zr = if fz > CBRT_EPSILON {
        fz.powi(3)
    } else {
        ((fz * 116.0) - 16.0) / KAPPA
    };

    [xr * 0.95047, yr, zr * 1.08883]
}

fn xyz_to_rgb(xyz: [f32; 3]) -> [u8; 3] {
    let x = xyz[0];
    let y = xyz[1];
    let z = xyz[2];

    let r = x * 3.2404541621141054 - y * 1.5371385127977166 - z * 0.4985314095560162;
    let g = x * -0.9692660305051868 + y * 1.8760108454466942 + z * 0.04155601753034984;
    let b = x * 0.05564343095911469 - y * 0.20402591351675387 + z * 1.0572251882231791;

    [xyz_to_rgb_map(r), xyz_to_rgb_map(g), xyz_to_rgb_map(b)]
}

#[inline]
fn xyz_to_rgb_map(c: f32) -> u8 {
    ((if c > 0.0031308 {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * c
    }) * 255.0)
        .round()
        .min(255.0)
        .max(0.0) as u8
}

/// Convenience function to map a slice of RGB values to Lab values in serial
///
/// # Example
/// ```
/// # extern crate lab;
/// # use lab::{Lab, rgbs_to_labs};
/// let rgbs = &[[0u8, 127, 127], [127, 0, 127], [255, 0, 0]];
/// let labs = lab::rgbs_to_labs(rgbs);
/// assert_eq!(labs, vec![
///     Lab { l: 47.8919, a: -28.683678, b: -8.42911 },
///     Lab { l: 29.52658, a: 58.595745, b: -36.281406 },
///     Lab { l: 53.240784, a: 80.09252, b: 67.203186 }
/// ]);
/// ```
#[inline]
pub fn rgbs_to_labs(rgbs: &[[u8; 3]]) -> Vec<Lab> {
    rgbs.iter().map(Lab::from_rgb).collect()
}

/// Convenience function to map a slice of Lab values to RGB values in serial
///
/// # Example
/// ```
/// # extern crate lab;
/// # use lab::{Lab, labs_to_rgbs};
/// let labs = &[
///     Lab { l: 91.11321, a: -48.08751, b: -14.131201 },
///     Lab { l: 60.32421, a: 98.23433, b: -60.824894 },
///     Lab { l: 97.13926, a: -21.553724, b: 94.47797 },
/// ];
/// let rgbs = lab::labs_to_rgbs(labs);
/// assert_eq!(rgbs, vec![[0u8, 255, 255], [255, 0, 255], [255, 255, 0]]);
/// ```
#[inline]
pub fn labs_to_rgbs(labs: &[Lab]) -> Vec<[u8; 3]> {
    labs.iter().map(Lab::to_rgb).collect()
}

impl Lab {
    // pub fn from_rgbs(rgbs: &[[u8; 3]]) -> Vec<Self> {
    //     #[cfg(target_arch = "x86_64")]
    //     {
    //         if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
    //             return simd::rgbs_to_labs(rgbs);
    //         }
    //     }
    //     rgbs_to_labs(rgbs)
    // }

    // pub fn to_rgbs(labs: &[Lab]) -> Vec<[u8; 3]> {
    //     #[cfg(target_arch = "x86_64")]
    //     {
    //         if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
    //             return simd::labs_to_rgbs(labs);
    //         }
    //     }
    //     labs_to_rgbs(labs)
    // }

    /// Constructs a new `Lab` from a three-element array of `u8`s
    ///
    /// # Examples
    ///
    /// ```
    /// let lab = lab::Lab::from_rgb(&[240, 33, 95]);
    /// assert_eq!(lab::Lab { l: 52.33686, a: 75.5516, b: 19.998878 }, lab);
    /// ```
    pub fn from_rgb(rgb: &[u8; 3]) -> Self {
        xyz_to_lab(rgb_to_xyz(rgb))
    }

    /// Constructs a new `Lab` from a four-element array of `u8`s
    ///
    /// The `Lab` struct does not store alpha channel information, so the last
    /// `u8` representing alpha is discarded. This convenience method exists
    /// in order to easily measure colors already stored in an RGBA array.
    ///
    /// # Examples
    ///
    /// ```
    /// let lab = lab::Lab::from_rgba(&[240, 33, 95, 255]);
    /// assert_eq!(lab::Lab { l: 52.33686, a: 75.5516, b: 19.998878 }, lab);
    /// ```
    pub fn from_rgba(rgba: &[u8; 4]) -> Self {
        Lab::from_rgb(&[rgba[0], rgba[1], rgba[2]])
    }

    /// Returns the `Lab`'s color in RGB, in a 3-element array.
    ///
    /// # Examples
    ///
    /// ```
    /// let lab = lab::Lab { l: 52.330193, a: 75.56704, b: 19.989174 };
    /// let rgb = lab.to_rgb();
    /// assert_eq!([240, 33, 95], rgb);
    /// ```
    pub fn to_rgb(&self) -> [u8; 3] {
        xyz_to_rgb(lab_to_xyz(&self))
    }

    /// Measures the perceptual distance between the colors of one `Lab`
    /// and an `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lab::Lab;
    /// let pink = Lab { l: 66.6348, a: 52.260696, b: 14.850557 };
    /// let websafe_pink = Lab { l: 64.2116, a: 62.519463, b: 2.8871894 };
    /// assert_eq!(254.23636, pink.squared_distance(&websafe_pink));
    /// ```
    pub fn squared_distance(&self, other: &Lab) -> f32 {
        (self.l - other.l).powi(2) + (self.a - other.a).powi(2) + (self.b - other.b).powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::{labs_to_rgbs, rgbs_to_labs, Lab};
    use rand;
    use rand::distributions::Standard;
    use rand::Rng;

    const PINK: Lab = Lab {
        l: 66.639084,
        a: 52.251457,
        b: 14.860654,
    };

    #[rustfmt::skip]
    static COLOURS: [([u8; 3], Lab); 17] = [
        ([253, 120, 138], PINK),

        ([127,   0,   0], Lab { l: 25.301395, a: 47.77433, b: 37.754025 }),
        ([  0, 127,   0], Lab { l: 45.87666, a: -51.40707, b: 49.615574 }),
        ([  0,   0, 127], Lab { l: 12.808655, a: 47.23452, b: -64.33745 }),
        ([  0, 127, 127], Lab { l: 47.8919, a: -28.683678, b: -8.42911 }),
        ([127,   0, 127], Lab { l: 29.52658, a: 58.595745, b: -36.281406 }),
        ([255,   0,   0], Lab { l: 53.240784, a: 80.09252, b: 67.203186 }),
        ([  0, 255,   0], Lab { l: 87.73472, a: -86.18272, b: 83.17931 }),
        ([  0,   0, 255], Lab { l: 32.29701, a: 79.187515, b: -107.86016 }),
        ([  0, 255, 255], Lab { l: 91.11321, a: -48.08751, b: -14.131201 }),
        ([255,   0, 255], Lab { l: 60.32421, a: 98.23433, b: -60.824894 }),
        ([255, 255,   0], Lab { l: 97.13926, a: -21.553724, b: 94.47797 }),

        ([  0,   0,   0], Lab { l: 0.0, a: 0.0, b: 0.0 }),
        ([ 64,  64,  64], Lab { l: 27.09341, a: 0.0, b: 0.0 }),
        ([127, 127, 127], Lab { l: 53.192772, a: 0.0, b: 0.0 }),
        ([196, 196, 196], Lab { l: 79.15698, a: 0.0, b: 0.0 }),
        ([255, 255, 255], Lab { l: 100.0, a: 0.0, b: 0.0 }),
    ];

    #[test]
    fn test_from_rgb() {
        for test in COLOURS.iter() {
            assert_eq!(test.1, Lab::from_rgb(&test.0));
            assert_eq!(
                test.1,
                Lab::from_rgba(&[test.0[0], test.0[1], test.0[2], 255])
            );
        }
    }

    #[test]
    fn test_to_rgb() {
        for test in COLOURS.iter() {
            assert_eq!(test.0, test.1.to_rgb());
        }
    }

    #[test]
    fn test_distance() {
        let ugly_websafe_pink = Lab {
            l: 64.2116,
            a: 62.519463,
            b: 2.8871894,
        };
        assert_eq!(PINK.squared_distance(&ugly_websafe_pink), 254.68846);
    }

    #[test]
    fn test_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Lab>();
    }

    #[test]
    fn test_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Lab>();
    }

    #[test]
    fn test_rgb_to_lab_to_rgb() {
        let rgbs: Vec<[u8; 3]> = {
            let rand_seed = [1u8; 32];
            let mut rng: rand::StdRng = rand::SeedableRng::from_seed(rand_seed);
            rng.sample_iter(&Standard).take(2048).collect()
        };
        let labs = rgbs_to_labs(&rgbs);
        let rgbs2 = labs_to_rgbs(&labs);
        assert_eq!(rgbs2, rgbs);
    }
}
