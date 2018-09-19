//! # Lab
//!
//! Tools for converting RGB colors to L\*a\*b\* measurements.

#![doc(html_root_url = "https://docs.rs/lab/0.4.4")]

#[derive(Debug, PartialEq, Copy, Clone, Default)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

fn rgb_to_xyz(rgb: &[u8; 3]) -> [f32; 3] {
    let r = rgb_to_xyz_map(rgb[0] as f32 / 255.0);
    let g = rgb_to_xyz_map(rgb[1] as f32 / 255.0);
    let b = rgb_to_xyz_map(rgb[2] as f32 / 255.0);

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

fn xyz_to_lab(xyz: [f32; 3]) -> Lab {
    let x = xyz_to_lab_map(xyz[0] / 95.047);
    let y = xyz_to_lab_map(xyz[1] / 100.0);
    let z = xyz_to_lab_map(xyz[2] / 108.883);

    Lab {
        l: (116.0 * y) - 16.0,
        a: 500.0 * (x - y),
        b: 200.0 * (y - z),
    }
}

#[inline]
fn xyz_to_lab_map(c: f32) -> f32 {
    if c > 0.008856 {
        c.powf(1.0/3.0)
    } else {
        (c * 7.787) + ( 16.0 / 116.0 )
    }
}

fn lab_to_xyz(lab: &Lab) -> [f32; 3] {
    let fy = (lab.l + 16.0) / 116.0;
    let fx = (lab.a / 500.0) + fy;
    let fz = fy - (lab.b / 200.0);
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
        if lab.l > 0.008856 * 903.3 {
            raised
        } else {
            lab.l / 903.3
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

fn xyz_to_rgb(xyz: [f32; 3]) -> [u8; 3] {
    let x = xyz[0] / 100.0;
    let y = xyz[1] / 100.0;
    let z = xyz[2] / 100.0;

    [
        xyz_to_rgb_map(x *  3.2406 + y * -1.5372 + z * -0.4986),
        xyz_to_rgb_map(x * -0.9689 + y *  1.8758 + z *  0.0415),
        xyz_to_rgb_map(x *  0.0557 + y * -0.2040 + z *  1.0570),
    ]
}

#[inline]
fn xyz_to_rgb_map(c: f32) -> u8 {
    ((if c > 0.0031308 {
         1.055 * c.powf(1.0/2.4) - 0.055
    } else {
         12.92 * c
    }) * 255.0).round().min(255.0).max(0.0) as u8
}

impl Lab {
    /// Constructs a new `Lab` from a three-element array of `u8`s
    ///
    /// # Examples
    ///
    /// ```
    /// let lab = lab::Lab::from_rgb(&[240, 33, 95]);
    /// assert_eq!(lab::Lab { l: 52.330193, a: 75.56704, b: 19.989174 }, lab);
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
    /// assert_eq!(lab::Lab { l: 52.330193, a: 75.56704, b: 19.989174 }, lab);
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
        (self.l - other.l).powi(2) +
        (self.a - other.a).powi(2) +
        (self.b - other.b).powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::Lab;

    const PINK: Lab = Lab { l: 66.6348, a: 52.260696, b: 14.850557 };

    static COLOURS: [([u8; 3], Lab); 17] = [
        ([253, 120, 138], PINK),

        ([127,   0,   0], Lab { l: 25.29668, a: 47.78436, b: 37.75621 }),
        ([  0, 127,   0], Lab { l: 45.878044, a: -51.40823, b: 49.616688 }),
        ([  0,   0, 127], Lab { l: 12.811981, a: 47.239967, b: -64.33954 }),
        ([  0, 127, 127], Lab { l: 47.893875, a: -28.678999, b: -8.433235 }),
        ([127,   0, 127], Lab { l: 29.524033, a: 58.60761, b: -36.292194 }),
        ([255,   0,   0], Lab { l: 53.23288, a: 80.10936, b: 67.22006 }),
        ([  0, 255,   0], Lab { l: 87.73704, a: -86.184654, b: 83.181175 }),
        ([  0,   0, 255], Lab { l: 32.302586, a: 79.19668, b: -107.863686 }),
        ([  0, 255, 255], Lab { l: 91.11652, a: -48.07961, b: -14.138126 }),
        ([255,   0, 255], Lab { l: 60.31993, a: 98.254234, b: -60.84299 }),
        ([255, 255,   0], Lab { l: 97.138245, a: -21.5559, b: 94.48248 }),

        // a and b in all of the below should be 0.0 but due to accumulation of
        // rounding errors, we end up with some non-zero values.
        ([  0,   0,   0], Lab { l: 0.0, a: 0.0, b: 0.0 }),
        ([ 64,  64,  64], Lab { l: 27.093414, a: 0.0019669533, b: -0.0038683414 }),
        ([127, 127, 127], Lab { l: 53.192772, a: 0.0031292439, b: -0.006210804 }),
        ([196, 196, 196], Lab { l: 79.15699, a: 0.0043213367, b: -0.008535385 }),
        ([255, 255, 255], Lab { l: 100.0, a: 0.0052452087, b: -0.010418892 }),
    ];

    #[test]
    fn test_from_rgb() {
        for test in COLOURS.iter() {
            assert_eq!(test.1, Lab::from_rgb(&test.0));
            assert_eq!(test.1,
                       Lab::from_rgba(&[test.0[0], test.0[1], test.0[2], 255]));
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
