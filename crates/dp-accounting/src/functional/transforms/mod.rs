//! Mechanism transforms
//!
//! Combinators that wrap a base mechanism and produce a new mechanism with
//! modified privacy characteristics.  Unlike amplification (which *reduces*
//! privacy cost via subsampling) or composition (which *accumulates* cost
//! over rounds), transforms reshape the mechanism itself — e.g. by folding
//! parallel queries into one equivalent mechanism.

pub mod adaclip;

pub use adaclip::{adaclip, AdaClip, AdaClipEvidence, AdaClipable, GaussianAdaClipEvidence};
