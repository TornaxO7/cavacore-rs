use std::{
    num::{NonZeroU32, NonZeroUsize},
    ops::Range,
};

use crate::{Channels, Error, SampleRate};

/// The recommended default noise reduction value.
/// Is automatically set if a new [`CavaBuilder`] is created.
pub const DEFAULT_NOISE_REDUCTION: f64 = 0.77;

#[derive(Debug, Clone)]
pub struct CavaBuilder {
    pub(crate) bars_per_channel: usize,
    pub(crate) sample_rate: u32,
    pub(crate) audio_channels: Channels,
    pub(crate) enable_autosens: bool,
    pub(crate) noise_reduction: f64,
    pub(crate) freq_range: Range<u32>,
}

impl CavaBuilder {
    pub fn bars_per_channel(mut self, bars_per_channel: NonZeroUsize) -> Self {
        self.bars_per_channel = usize::from(bars_per_channel);
        self
    }

    pub fn sample_rate(mut self, sample_rate: SampleRate) -> Self {
        self.sample_rate = u32::from(sample_rate);
        self
    }

    pub fn audio_channels(mut self, audio_channels: Channels) -> Self {
        self.audio_channels = audio_channels;
        self
    }

    pub fn enable_autosens(mut self, enable_autosens: bool) -> Self {
        self.enable_autosens = enable_autosens;
        self
    }

    /// Adjust noise reduciton filters. Has to be within the range `0..=1`.
    ///
    /// The raw visualization is very noisy, this factor adjusts the integral
    /// and gravity filters inside cavacore to keep the signal smooth:
    /// `1` will be very slow and smooth, `0` will be fast but noisy.
    pub fn noise_reduction(mut self, noise_reduction: f64) -> Self {
        self.noise_reduction = noise_reduction;
        self
    }

    pub fn frequency_range(mut self, freq_range: Range<NonZeroU32>) -> Self {
        let start = u32::from(freq_range.start);
        let end = u32::from(freq_range.end);

        self.freq_range = Range { start, end };
        self
    }

    pub fn sanity_check(&self) -> Result<(), Vec<Error>> {
        let mut errors = Vec::new();
        let treble_buffer_size = self.compute_treble_buffer_size();

        if self.bars_per_channel > treble_buffer_size / 2 + 1 {
            errors.push(Error::TooHighAmountBars {
                amount_bars: self.bars_per_channel,
                sample_rate: self.sample_rate,
                max_amount_bars: treble_buffer_size / 2 + 1,
            });
        }

        if self.freq_range.is_empty() {
            errors.push(Error::EmptyFreqRange {
                start: self.freq_range.start,
                end: self.freq_range.end,
            });
        }

        if self.freq_range.end > self.sample_rate / 2 {
            errors.push(Error::NyquistIssue {
                freq: self.freq_range.end,
                max_freq: self.sample_rate / 2,
            });
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(())
    }

    pub(crate) fn compute_treble_buffer_size(&self) -> usize {
        let factor = match self.sample_rate {
            ..=08_125 => 1,
            ..=16_250 => 2,
            ..=32_500 => 4,
            ..=75_000 => 8,
            ..=150_000 => 16,
            ..=300_000 => 32,
            _ => 64,
        };

        factor * 128
    }
}

impl Default for CavaBuilder {
    fn default() -> Self {
        Self {
            bars_per_channel: 32,
            sample_rate: 44_100,
            audio_channels: Channels::Stereo,
            enable_autosens: true,
            noise_reduction: 0.77,
            freq_range: 50..10_000,
        }
    }
}
