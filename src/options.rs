use std::ops::Range;

pub type Hz = u32;

#[derive(Debug, Clone)]
pub struct CavaOptions {
    pub enable_autosens: bool,
    pub noise_reduction: f64,
    pub freq_range: Range<Hz>,
}

impl Default for CavaOptions {
    fn default() -> Self {
        Self {
            enable_autosens: true,
            noise_reduction: 0.77,
            freq_range: 50..10_000,
        }
    }
}
