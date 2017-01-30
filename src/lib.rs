#[derive(Debug, PartialEq)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Lab {
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        let (x, y, z) = {
            let r = (r as f32) / 255.0;
            let g = (g as f32) / 255.0;
            let b = (b as f32) / 255.0;
            let _rgb: Vec<f32> = [r, g, b].into_iter().map(|&c| {
                (if c > 0.04045 {
                    ((c + 0.055) / 1.055).powf(2.4)
                } else {
                    c / 12.92
                }) * 100.0
            }).collect();

            (
                _rgb[0]*0.4124 + _rgb[1]*0.3576 + _rgb[2]*0.1805,
                _rgb[0]*0.2126 + _rgb[1]*0.7152 + _rgb[2]*0.0722,
                _rgb[0]*0.0193 + _rgb[1]*0.1192 + _rgb[2]*0.9505,
            )
        };

        let (l, a, b) = {
            let _x = x / 95.047;
            let _y = y / 100.0;
            let _z = z / 108.883;

            let _xyz: Vec<f32> = [_x, _y, _z].into_iter().map(|&c| {
                if c > 0.008856 {
                    c.powf(1.0/3.0)
                } else {
                    (c * 7.787) + ( 16.0 / 116.0 )
                }
            }).collect();

            (
                (116.0 * _xyz[1]) - 16.0,
                500.0 * (_xyz[0] - _xyz[1]),
                200.0 * (_xyz[1] - _xyz[2]),
            )
        };

        Lab { l: l, a: a, b: b }
    }

    pub fn squared_distance(&self, other: &Lab) -> f32 {
        (self.l - other.l).powf(2.0) +
        (self.a - other.a).powf(2.0) +
        (self.b - other.b).powf(2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Lab;

    #[test]
    fn test_from_rgb() {
        let rgb: [u8; 3] = [253, 120, 138];
        assert_eq!(
            Lab::from_rgb(rgb[0], rgb[1], rgb[2]),
            Lab { l: 66.6348, a: 52.260696, b: 14.850557 }
        );
    }
}
