//! Experimental functions for parallel conversion using AVX and SSE4.1
//!
//! This module is conditionally compiled by the cfg gate
//! `#[cfg(target_arch = "x86_64")]`

mod labs_to_rgbs;
mod rgbs_to_labs;
mod math;

pub use self::labs_to_rgbs::{labs_to_rgbs, labs_to_rgbs_chunk};
pub use self::rgbs_to_labs::{rgbs_to_labs, rgbs_to_labs_chunk};
use super::{Lab, CBRT_EPSILON, EPSILON, KAPPA};
