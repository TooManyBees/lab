# Rust library for converting RGB colors to the CIE-L\*a\*b\* color space

```rust
extern crate lab;
use lab::Lab;

let pink_in_lab = Lab::from_rgb(&[253, 120, 138]);
// Lab { l: 66.639084, a: 52.251457, b: 14.8606415 }
```

```rust
extern crate lab;
extern crate image;

use lab::Lab;
use image::Rgba;

let pixel: Rgba<u8> = Rgba { data: [253, 120, 138, 255] };
let lab = Lab::from_rgba(&pixel.data);
// Lab { l: 66.639084, a: 52.251457, b: 14.8606415 }
```
