use std::{num::NonZeroUsize, ops::Range};

use crate::{AudioData, Cava, CavaOpts, Channels, Error};

impl Cava {
    pub fn new(opts: CavaOpts) -> Result<Self, Vec<Error>> {
        let bars_per_channel = usize::from(opts.bars_per_channel);
        let sample_rate = u32::from(opts.sample_rate);
        let freq_range = Range {
            start: u32::from(opts.frequency_range.start),
            end: u32::from(opts.frequency_range.end),
        };

        let mut errors = Vec::new();

        let treble_buffer_size = Self::compute_treble_buffer_size(sample_rate);
        if bars_per_channel > treble_buffer_size / 2 + 1 {
            errors.push(Error::TooHighAmountBars {
                amount_bars: bars_per_channel,
                sample_rate,
                max_amount_bars: treble_buffer_size / 2 + 1,
            });
        }

        if freq_range.is_empty() {
            errors.push(Error::EmptyFreqRange {
                start: freq_range.start,
                end: freq_range.end,
            });
        }

        if freq_range.end > sample_rate / 2 {
            errors.push(Error::NyquistIssue {
                freq: freq_range.end,
                max_freq: sample_rate / 2,
            });
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(Self::new_inner(
            opts.bars_per_channel,
            sample_rate,
            opts.audio_channels,
            opts.enable_autosens,
            opts.noise_reduction,
            freq_range,
        ))
    }

    fn new_inner(
        bars_per_channel: NonZeroUsize,
        sample_rate: u32,
        audio_channels: Channels,
        enable_autosens: bool,
        noise_reduction: f64,
        freq_range: Range<u32>,
    ) -> Self {
        let bars_per_channel_usize = usize::from(bars_per_channel);
        let init_sens = true;
        let sens = 1.;
        let framerate = 75.;
        let frame_skip = 1.;

        let treble_buffer_size = Self::compute_treble_buffer_size(sample_rate);
        let (left, right): (AudioData<f64>, Option<AudioData<f64>>) = {
            let left = AudioData::new(treble_buffer_size);
            let right = match audio_channels {
                Channels::Mono => None,
                Channels::Stereo => Some(AudioData::new(treble_buffer_size)),
            };
            (left, right)
        };

        let bass_hann_window = compute_hann_window(left.in_bass.len());
        let mid_hann_window = compute_hann_window(left.in_mid.len());
        let treble_hann_window = compute_hann_window(left.in_treble.len());

        let input_buffer =
            vec![0.0; left.in_bass.len() * audio_channels as usize].into_boxed_slice();

        let buffer_lower_cut_off = vec![0; bars_per_channel_usize + 1];
        let buffer_upper_cut_off = vec![0; bars_per_channel_usize + 1];
        let eq = vec![0.; bars_per_channel_usize + 1];

        let total_amount_bars = bars_per_channel_usize * audio_channels as usize;
        let cava_fall = vec![0.0; total_amount_bars].into_boxed_slice();
        let cava_mem = vec![0.0; total_amount_bars].into_boxed_slice();
        let cava_peak = vec![0.0; total_amount_bars].into_boxed_slice();
        let prev_cava_out = vec![0.0; total_amount_bars].into_boxed_slice();

        let bass_cut_off_bar = -1;
        let treble_cut_off_bar = -1;

        let mut cava = Cava {
            bars_per_channel: bars_per_channel_usize,
            sample_rate,
            enable_autosens,
            init_sens,
            sens,
            framerate,
            frame_skip,
            noise_reduction,
            freq_range,
            left,
            right,
            bass_hann_window,
            mid_hann_window,
            treble_hann_window,
            input_buffer,
            buffer_lower_cut_off,
            buffer_upper_cut_off,
            eq,
            bass_cut_off_bar,
            treble_cut_off_bar,
            cava_fall,
            cava_mem,
            cava_peak,
            prev_cava_out,
        };

        cava.set_bars(bars_per_channel).unwrap();
        cava
    }

    pub(crate) fn compute_treble_buffer_size(sample_rate: u32) -> usize {
        let factor = if sample_rate <= 8_125 {
            1
        } else if sample_rate <= 16_250 {
            2
        } else if sample_rate <= 32_500 {
            4
        } else if sample_rate <= 75_000 {
            8
        } else if sample_rate <= 150_000 {
            16
        } else if sample_rate <= 300_000 {
            32
        } else {
            64
        };

        factor * 128
    }
}

fn compute_hann_window(buffer_size: usize) -> Box<[f64]> {
    let mut hann_window = Vec::with_capacity(buffer_size);

    for i in 0..buffer_size {
        let multiplier =
            0.5 * (1. - (2. * std::f64::consts::PI * i as f64 / (buffer_size as f64 - 1.)).cos());
        hann_window.push(multiplier);
    }

    hann_window.into_boxed_slice()
}
