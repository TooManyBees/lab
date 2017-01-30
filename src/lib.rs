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

    pub fn to_rgb(&self) -> (u8, u8, u8) {
        let (x, y, z) = {
            let y = (self.l + 16.0) / 116.0;
            let x = (self.a / 500.0) + y;
            let z = y - (self.b / 200.0);

            let xyz: Vec<f32> = [x, y, z].into_iter().map(|&c| {
                let raised = c.powf(3.0);
                if raised > 0.008856 {
                    raised
                } else {
                    (c - 16.0) / 116.0 / 7.787
                }
            }).collect();

            (
                xyz[0] * 95.047,
                xyz[1] * 100.0,
                xyz[2] * 108.883,
            )
        };

        let (r, g, b) = {
            let x = x / 100.0;
            let y = y / 100.0;
            let z = z / 100.0;

            let r = x *  3.2406 + y * -1.5372 + z * -0.4986;
            let g = x * -0.9689 + y *  1.8758 + z *  0.0415;
            let b = x *  0.0557 + y * -0.2040 + z *  1.0570;

            let _rgb: Vec<f32> = [r, g, b].into_iter().map(|&c| {
                (if c > 0.0031308 {
                     1.055 * c.powf(1.0/2.4) - 0.055
                } else {
                     12.92 * c
                } * 255.0).round()
            }).collect();

            (_rgb[0] as u8, _rgb[1] as u8, _rgb[2] as u8)
        };

        (r, g, b)
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

    #[test]
    fn test_to_rgb() {
        let lab = Lab { l: 66.6348, a: 52.260696, b: 14.850557 };
        assert_eq!(
            lab.to_rgb(),
            (253, 120, 138)
        );
    }
}
