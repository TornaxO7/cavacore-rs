//! A rust rewrite of the [core processing engine] of cava.
//!
//! [core processing engine]: https://github.com/karlstav/cava/blob/master/CAVACORE.md
use std::{
    num::{NonZeroU32, NonZeroUsize},
    ops::Range,
    sync::Arc,
};

use bounded_integer::BoundedU32;
use realfft::{num_complex::Complex, FftNum, RealFftPlanner, RealToComplex};

mod error;
mod init;
mod setter;

pub use error::Error;

/// A helper type to restrict the sample rate within the given range.
pub type SampleRate = BoundedU32<1, 384_000>;

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Channels {
    Mono = 1,
    Stereo,
}

#[derive(Debug)]
pub struct CavaOpts {
    pub bars_per_channel: NonZeroUsize,
    pub sample_rate: SampleRate,
    pub audio_channels: Channels,
    pub enable_autosens: bool,
    pub noise_reduction: f64,
    pub frequency_range: Range<NonZeroU32>,
}

impl Default for CavaOpts {
    fn default() -> Self {
        Self {
            bars_per_channel: NonZeroUsize::new(32).unwrap(),
            sample_rate: SampleRate::new(44_100).unwrap(),
            audio_channels: Channels::Stereo,
            enable_autosens: true,
            noise_reduction: 0.77,
            frequency_range: NonZeroU32::new(50).unwrap()..NonZeroU32::new(10_000).unwrap(),
        }
    }
}

pub struct Cava {
    bars_per_channel: usize,
    sample_rate: u32,
    enable_autosens: bool,
    init_sens: bool,
    sens: f64,
    framerate: f64,
    frame_skip: f64,
    noise_reduction: f64,
    freq_range: Range<u32>,

    left: AudioData<f64>,
    right: Option<AudioData<f64>>,

    bass_hann_window: Box<[f64]>,
    mid_hann_window: Box<[f64]>,
    treble_hann_window: Box<[f64]>,

    input_buffer: Box<[f64]>,
    buffer_lower_cut_off: Vec<u32>,
    buffer_upper_cut_off: Vec<u32>,
    eq: Vec<f64>,

    bass_cut_off_bar: i32,
    treble_cut_off_bar: i32,

    cava_fall: Box<[f64]>,
    cava_mem: Box<[f64]>,
    cava_peak: Box<[f64]>,
    prev_cava_out: Box<[f64]>,
}

impl Cava {
    pub fn make_output(&self) -> Box<[f64]> {
        let amount_channels = if self.right.is_some() { 2 } else { 1 };
        let total_amount_bars = self.bars_per_channel * amount_channels;

        vec![0.; total_amount_bars].into_boxed_slice()
    }
}

impl Cava {
    pub fn execute(&mut self, input: &[f64], output: &mut [f64]) {
        if input.is_empty() {
            self.frame_skip += 1.;
        } else {
            self.framerate -= self.framerate / 64.;

            let amount_audio_channels = if self.right.is_some() { 2. } else { 1. };

            self.framerate += (self.sample_rate as f64 * amount_audio_channels * self.frame_skip)
                / input.len() as f64
                / 64.;

            self.frame_skip = 1.;

            let input_buffer_len = self.input_buffer.len();
            self.input_buffer
                .copy_within(..input_buffer_len - input.len(), input.len());
            self.input_buffer[..input.len()].copy_from_slice(input);
        }

        // fill the bass, mid and treble buffers
        if let Some(right) = self.right.as_mut() {
            // bass
            for i in 0..self.left.in_bass.len() {
                right.in_bass[i] = self.input_buffer[i * 2] * self.bass_hann_window[i];
                self.left.in_bass[i] = self.input_buffer[i * 2 + 1] * self.bass_hann_window[i];
            }

            // mid
            for i in 0..self.left.in_mid.len() {
                right.in_mid[i] = self.input_buffer[i * 2] * self.mid_hann_window[i];
                self.left.in_mid[i] = self.input_buffer[i * 2 + 1] * self.mid_hann_window[i];
            }

            // treble
            for i in 0..self.left.in_treble.len() {
                right.in_treble[i] = self.input_buffer[i * 2] * self.treble_hann_window[i];
                self.left.in_treble[i] = self.input_buffer[i * 2 + 1] * self.treble_hann_window[i];
            }

            right
                .p_bass
                .process(&mut right.in_bass, &mut right.out_bass)
                .unwrap();
            right
                .p_mid
                .process(&mut right.in_mid, &mut right.out_mid)
                .unwrap();
            right
                .p_treble
                .process(&mut right.in_treble, &mut right.out_treble)
                .unwrap();
        } else {
            // bass
            for i in 0..self.left.in_bass.len() {
                self.left.in_bass[i] = self.input_buffer[i] * self.bass_hann_window[i];
            }

            // mid
            for i in 0..self.left.in_mid.len() {
                self.left.in_mid[i] = self.input_buffer[i] * self.mid_hann_window[i];
            }

            // treble
            for i in 0..self.left.in_treble.len() {
                self.left.in_treble[i] = self.input_buffer[i] * self.treble_hann_window[i];
            }
        }

        // fft goes brrrrrrr
        self.left
            .p_bass
            .process(&mut self.left.in_bass, &mut self.left.out_bass)
            .unwrap();
        self.left
            .p_mid
            .process(&mut self.left.in_mid, &mut self.left.out_mid)
            .unwrap();
        self.left
            .p_treble
            .process(&mut self.left.in_treble, &mut self.left.out_treble)
            .unwrap();

        // separate frequency bands
        for n in 0..self.bars_per_channel {
            let mut tmp_l = 0.;
            let mut tmp_r = 0.;

            // add upp FFT values within bands
            for i in self.buffer_lower_cut_off[n] as usize..=self.buffer_upper_cut_off[n] as usize {
                if n <= self.bass_cut_off_bar as usize {
                    tmp_l += self.left.out_bass[i].norm();
                    if let Some(right) = &self.right {
                        tmp_r += right.out_bass[i].norm();
                    }
                } else if (self.bass_cut_off_bar as usize..=self.treble_cut_off_bar as usize)
                    .contains(&n)
                {
                    tmp_l += self.left.out_mid[i].norm();
                    if let Some(right) = &self.right {
                        tmp_r += right.out_mid[i].norm();
                    }
                } else if (self.treble_cut_off_bar as usize) < n {
                    tmp_l += self.left.out_treble[i].norm();
                    if let Some(right) = &self.right {
                        tmp_r += right.out_treble[i].norm();
                    }
                }
            }

            // getting average multiply with eq
            tmp_l /= self.buffer_upper_cut_off[n] as f64 - self.buffer_lower_cut_off[n] as f64 + 1.;
            tmp_l *= self.eq[n];
            output[n] = tmp_l;

            if self.right.is_some() {
                tmp_r /=
                    self.buffer_upper_cut_off[n] as f64 - self.buffer_lower_cut_off[n] as f64 + 1.;
                tmp_r *= self.eq[n];
                output[n + self.bars_per_channel] = tmp_r;
            }
        }

        // applying sens or getting max value
        if self.enable_autosens {
            for val in output.iter_mut() {
                *val *= self.sens;
            }
        }

        // process [smoothing]
        let mut is_overshooting = false;
        // THONK: eeeh... what happens if noise_reduction equals zero??
        let gravity_mod = {
            let gravity_mod = (60. / self.framerate).powf(2.5) * 1.54 / self.noise_reduction;

            gravity_mod.max(1.)
        };

        let amount_channels = if self.right.is_some() { 2 } else { 1 };
        for (n, out) in output
            .iter_mut()
            .enumerate()
            .take(self.bars_per_channel * amount_channels)
        {
            // [smoothing]: falloff
            if *out < self.prev_cava_out[n] && self.noise_reduction > 0.1 {
                *out = self.cava_peak[n] * (1. - (self.cava_fall[n].powf(2.) * gravity_mod));

                if *out < 0.0 {
                    *out = 0.0;
                }
                self.cava_fall[n] += 0.028;
            } else {
                self.cava_peak[n] = *out;
                self.cava_fall[n] = 0.;
            }
            self.prev_cava_out[n] = *out;

            // [smoothing]: integral
            *out += self.cava_mem[n] * self.noise_reduction;
            self.cava_mem[n] = *out;
            if self.enable_autosens {
                // check if we overshoot target height
                if *out > 1. {
                    is_overshooting = true;
                }
            }
        }

        // calculating automatic sense adjustment
        if self.enable_autosens {
            if is_overshooting {
                self.sens *= 0.98;
                self.init_sens = false;
            } else {
                let is_silent = input.iter().all(|&v| v == 0.);
                if is_silent {
                    self.sens *= 1.002;
                    if self.init_sens {
                        self.sens *= 1.1;
                    }
                }
            }
        }
    }
}

struct AudioData<F: FftNum> {
    p_bass: Arc<dyn RealToComplex<F>>,
    p_mid: Arc<dyn RealToComplex<F>>,
    p_treble: Arc<dyn RealToComplex<F>>,

    out_bass: Vec<Complex<F>>,
    out_mid: Vec<Complex<F>>,
    out_treble: Vec<Complex<F>>,

    in_bass: Vec<F>,
    in_mid: Vec<F>,
    in_treble: Vec<F>,
}

impl<F: FftNum> AudioData<F> {
    pub fn new(treble_buffer_size: usize) -> Self {
        let mut planner = RealFftPlanner::new();

        let bass = planner.plan_fft_forward(treble_buffer_size * 8);
        let mid = planner.plan_fft_forward(treble_buffer_size * 4);
        let treble = planner.plan_fft_forward(treble_buffer_size);

        let out_bass = bass.make_output_vec();
        let out_mid = mid.make_output_vec();
        let out_treble = treble.make_output_vec();

        let in_bass = bass.make_input_vec();
        let in_mid = mid.make_input_vec();
        let in_treble = treble.make_input_vec();

        Self {
            p_bass: bass,
            p_mid: mid,
            p_treble: treble,

            out_bass,
            out_mid,
            out_treble,

            in_bass,
            in_mid,
            in_treble,
        }
    }
}

#[cfg(test)]
mod tests {
    mod audio_data {
        use crate::AudioData;

        #[test]
        fn buffer_size() {
            let treble_buffer_size = 128 * 4;

            let audio_data: AudioData<f32> = AudioData::new(treble_buffer_size);

            assert_eq!(audio_data.in_bass.len(), treble_buffer_size * 8);
            assert_eq!(audio_data.in_mid.len(), treble_buffer_size * 4);
            assert_eq!(audio_data.in_treble.len(), treble_buffer_size);
        }
    }
}
