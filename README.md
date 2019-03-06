# Rust library for converting RGB colors to the CIE-L\*a\*b\* color space
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/TooManyBees/lab?branch=master&svg=true)](https://ci.appveyor.com/project/TooManyBees/lab)

```rust
extern crate lab;
use lab::Lab;

let pink_in_lab = Lab::from_rgb(&[253, 120, 138]);
// Lab { l: 66.639084, a: 52.251457, b: 14.860654 }
```

```rust
extern crate lab;
extern crate image;

use lab::Lab;
use image::Rgba;

let pixel: Rgba<u8> = Rgba { data: [253, 120, 138, 255] };
let lab = Lab::from_rgba(&pixel.data);
// Lab { l: 66.639084, a: 52.251457, b: 14.860654 }
```

# Experimental SIMD functions

The `lab::simd` module is compiled for the `x86_64` cpu architecture. If the
current cpu can run AVX and SSE 4.1 operations, it can make use of the exported
functions.

```rust
extern crate lab;
use lab::Lab;
#[cfg(target_arch = "x86_64")]
use lab::simd;

fn convert_rgbs(rgbs: &[[u8; 3]]) -> Vec<Lab> {
  // It's boilerplate, but it's also experimental. So.
  #[cfg(target_arch = "x86_64")]
  {
      if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") {
          return simd::rgbs_to_labs(rgbs);
      }
  }
  rgbs.iter().map(Lab::from_rgb).collect()
}
```

Performance increase over a serial map is wildly variable across CPUs, suggesting that
there are still some optimizations to perform. A 2013 Macbook Air sees a \~25% decrease in benchmark times converting Labs to RGBs, and a \~40% decrease converting RGBs to Labs.
Meanwhile a 6-core desktop computer sees near perfect 8x speedup converting Labs to RGBs,
but a <10% improvement converting RGBs to Labs. Clearly it is a work in progress.

# Minimum Rust version
Lab 0.7.0 requires Rust >= 1.31.0 for the [chunks_exact](https://doc.rust-lang.org/std/primitive.slice.html#method.chunks_exact) slice method

Lab 0.6.0 can build as far back as Rust 1.13.0. Testing releases gets pretty tedious earlier than that.
