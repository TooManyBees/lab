//! # Lab
//!
//! Tools for converting RGB colors to L\*a\*b\* measurements.

#![doc(html_root_url = "https://docs.rs/lab/0.4.4")]

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate lazy_static;
#[cfg(test)]
extern crate rand;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
pub use avx::rgbs_to_labs as rgbs_to_labs_avx;

#[derive(Debug, PartialEq, Copy, Clone, Default)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

fn rgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb_to_xyz_map(rgb[0] / 255.0);
    let g = rgb_to_xyz_map(rgb[1] / 255.0);
    let b = rgb_to_xyz_map(rgb[2] / 255.0);

    [
        r*0.4124 + g*0.3576 + b*0.1805,
        r*0.2126 + g*0.7152 + b*0.0722,
        r*0.0193 + g*0.1192 + b*0.9505,
    ]
}

#[inline]
fn rgb_to_xyz_map(c: f32) -> f32 {
    (if c > 0.04045 {
        ((c + 0.055) / 1.055).powf(2.4)
    } else {
        c / 12.92
    }) * 100.0
}

fn xyz_to_lab(xyz: [f32; 3]) -> [f32; 3] {
    let x = xyz_to_lab_map(xyz[0] / 95.047);
    let y = xyz_to_lab_map(xyz[1] / 100.0);
    let z = xyz_to_lab_map(xyz[2] / 108.883);

    [
        (116.0 * y) - 16.0,
        500.0 * (x - y),
        200.0 * (y - z),
    ]
}

#[inline]
fn xyz_to_lab_map(c: f32) -> f32 {
    if c > 0.008856 {
        c.powf(1.0/3.0)
    } else {
        (c * 7.787) + ( 16.0 / 116.0 )
    }
}

fn lab_to_xyz(lab: [f32; 3]) -> [f32; 3] {
    let l = lab[0];
    let a = lab[1];
    let b = lab[2];
    let fy = (l + 16.0) / 116.0;
    let fx = (a / 500.0) + fy;
    let fz = fy - (b / 200.0);
    let xr = {
        let raised = fx.powi(3);
        if raised > 0.008856 {
            raised
        } else {
            ((fx * 116.0) - 16.0) / 903.3
        }
    };
    let yr = {
        let raised = fy.powi(3);
        if l > 0.008856 * 903.3 {
            raised
        } else {
            l / 903.3
        }
    };
    let zr = {
        let raised = fz.powi(3);
        if raised > 0.008856 {
            raised
        } else {
            ((fz * 116.0) - 16.0) / 903.3
        }
    };

    [
        xr * 95.047,
        yr * 100.0,
        zr * 108.883,
    ]
}

fn xyz_to_rgb(xyz: [f32; 3]) -> [f32; 3] {
    let x = xyz[0] / 100.0;
    let y = xyz[1] / 100.0;
    let z = xyz[2] / 100.0;

    let r = xyz_to_rgb_map(x *  3.2406 + y * -1.5372 + z * -0.4986);
    let g = xyz_to_rgb_map(x * -0.9689 + y *  1.8758 + z *  0.0415);
    let b = xyz_to_rgb_map(x *  0.0557 + y * -0.2040 + z *  1.0570);

    [r, g, b]
}

#[inline]
fn xyz_to_rgb_map(c: f32) -> f32 {
    (if c > 0.0031308 {
        1.055 * c.powf(1.0/2.4) - 0.055
    } else {
        12.92 * c
    }) * 255.0
}

#[inline]
pub fn rgbs_to_labs(rgbs: &[[u8; 3]]) -> Vec<Lab> {
    rgbs.iter().map(Lab::from_rgb).collect()
}

impl Lab {

    pub fn from_rgbs(rgbs: &[[u8; 3]]) -> Vec<Self> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
                return unsafe { rgbs_to_labs_avx(rgbs) };
            }
        }
        rgbs_to_labs(rgbs)
    }

    /// Constructs a new `Lab` from a three-element array of `u8`s
    ///
    /// # Examples
    ///
    /// ```
    /// # use lab::Lab;
    /// let lab = Lab::from_rgb(&[240, 33, 95]);
    /// // Lab { l: 66.6348, a: 52.260696, b: 14.850557 }
    /// ```
    pub fn from_rgb(rgb: &[u8; 3]) -> Self {
        let xyz = rgb_to_xyz([rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]);
        let lab = xyz_to_lab(xyz);
        Lab {
            l: lab[0],
            a: lab[1],
            b: lab[2],
        }
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
    /// # use lab::Lab;
    /// let lab = Lab::from_rgba(&[240, 33, 95, 255]);
    /// // Lab { l: 66.6348, a: 52.260696, b: 14.850557 }
    /// ```
    pub fn from_rgba(rgba: &[u8; 4]) -> Self {
        Lab::from_rgb(&[rgba[0], rgba[1], rgba[2]])
    }

    /// Returns the `Lab`'s color in RGB, in a 3-element array.
    pub fn to_rgb(&self) -> [u8; 3] {
        let xyz = lab_to_xyz([self.l, self.a, self.b]);
        let rgb = xyz_to_rgb(xyz);
        [
            rgb[0].round().min(255.).max(0.) as u8,
            rgb[1].round().min(255.).max(0.) as u8,
            rgb[2].round().min(255.).max(0.) as u8,
        ]
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
    /// let dist = pink.squared_distance(&websafe_pink);
    /// // 254.23636
    /// ```
    pub fn squared_distance(&self, other: &Lab) -> f32 {
        (self.l - other.l).powi(2) +
        (self.a - other.a).powi(2) +
        (self.b - other.b).powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::Lab;

    const PINK: Lab = Lab { l: 66.6348, a: 52.260696, b: 14.850557 };

    #[test]
    fn test_from_rgb() {
        let rgb: [u8; 3] = [253, 120, 138];
        assert_eq!(
            Lab::from_rgb(&rgb),
            PINK
        );
    }

    #[test]
    fn test_to_rgb() {
        assert_eq!(
            PINK.to_rgb(),
            [253, 120, 138]
        );
    }

    #[test]
    fn test_distance() {
        let ugly_websafe_pink = Lab { l: 64.2116, a: 62.519463, b: 2.8871894 };
        assert_eq!(PINK.squared_distance(&ugly_websafe_pink), 254.23636);
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
}
