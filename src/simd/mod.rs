//! Experimental functions for parallel conversion using AVX and SSE4.1
//!
//! This module is conditionally compiled by the cfg gate
//! `#[cfg(target_arch = "x86_64")]`

#[cfg(test)]
mod approx_impl;
mod labs_to_rgbs;
mod math;
mod rgbs_to_labs;

pub use self::labs_to_rgbs::{labs_to_rgb_bytes, labs_to_rgbs};
pub use self::rgbs_to_labs::{
    abgr_bytes_to_labs, argb_bytes_to_labs, bgr_bytes_to_labs, bgra_bytes_to_labs,
    rgb_bytes_to_labs, rgba_bytes_to_labs, rgbs_to_labs,
};
