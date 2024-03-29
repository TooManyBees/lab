# 0.11.0

* bump `pretty_assertions` (dev dependency) crate from version 0.6 to 0.7
* bump `approx` (dev dependency) crate from version 0.3 to 0.5

# 0.10.0

* bump `rand` crate from version 0.5 to 0.8
* fix a potential edge case error. Per @mina86:
  > the `rgb_to_xyz_map` function uses threshold equal 10.  This is fine
  > if the argument is an integer in 0–255 range (255 * 0.0404 = 10.302)
  > which the function was originally designed to handle.  However, it
  > has since been changed to accept floating point numbers in said
  > range.  As a result, if a colour component is, say, 10.1, the
  > function will use the wrong gamma expansion formula.

# 0.9.0

* add `LCh` data struct - @mina86

# 0.8.2

* No code changes
* Includes Readme and license files in published crate

# 0.8.1

* Fix a bug in AVX2 path of `labs_to_rgb_bytes`

# 0.8.0

* Speed up AVX2 code path, primarily by bringing in an AVX2 impl of `powf`,
  `log`, and `exp`
* Remove `simd` as a public module
* `rgbs_to_labs` and `labs_to_rgbs`: uses AVX2 code paths when compiled on a
  target that supports it
* `rgb_bytes_to_labs` and `labs_to_rgb_bytes`: works with RGB triples in a flat
  `&[u8]` instead of `&[[u8; 3]]`

# 0.7.0

* add convenience methods for converting slices of labs/rgbs
* add experimental `simd` module with functions using avx and sse4.1 operations

# 0.6.0

* More accurate FP calculations - @mina86

# 0.5.0

* Update `rand` crate from version 0.3 to 0.5

# 0.4.4

Current version when `CHANGELOG.md` was created.
