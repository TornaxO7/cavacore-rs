use std::{
    f64::consts::PI,
    num::{NonZeroU32, NonZeroUsize},
    ops::Range,
    sync::Arc,
};

use bounded_integer::BoundedU32;
use realfft::{num_complex::Complex, FftNum, RealFftPlanner, RealToComplex};

pub use error::Error;

mod error;

/// A helper type to restrict the sample rate within the given range.
pub type SampleRate = BoundedU32<1, 384_000>;

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Channels {
    Mono = 1,
    Stereo,
}

pub struct Cava {
    bars_per_channel: usize,
    audio_channels: Channels,
    sample_rate: u32,
    bass_cut_off_bar: i32,
    treble_cut_off_bar: i32,
    init_sens: bool,
    enable_autosens: bool,
    frame_skip: i32,

    sens: f64,
    framerate: f64,
    noise_reduction: f64,

    left: AudioData<f64>,
    right: AudioData<f64>,

    bass_multiplier: Vec<f64>,
    mid_multiplier: Vec<f64>,
    treble_multiplier: Vec<f64>,

    prev_cava_out: Vec<f64>,
    input_buffer: Vec<f64>,
    cava_peak: Vec<f64>,
    cava_mem: Vec<f64>,
    cava_out: Vec<f64>,

    eq: Vec<f64>,

    cut_off_frequency: Vec<f64>,
    fft_buffer_lower_cut_off: Vec<usize>,
    fft_buffer_upper_cut_off: Vec<usize>,
    cava_fall: Vec<f64>,
}

impl Cava {
    pub fn new(
        bars_per_channel: NonZeroUsize,
        sample_rate: SampleRate,
        audio_channels: Channels,
        enable_autosens: bool,
        noise_reduction: f64,
        freq_range: Range<NonZeroU32>,
    ) -> Result<Self, Error> {
        let bars_per_channel = usize::from(bars_per_channel);
        let sample_rate = u32::from(sample_rate);
        let freq_range = Range {
            start: u32::from(freq_range.start),
            end: u32::from(freq_range.end),
        };

        let treble_in_buf_size: usize = {
            let factor = match sample_rate {
                ..=8124 => 1,
                8125..=16250 => 2,
                16251..=32500 => 4,
                32501..=75000 => 8,
                75001..=150000 => 16,
                150001..=300000 => 32,
                _ => 64,
            };

            128 * factor
        };
        let mid_in_buf_size = treble_in_buf_size * 4;
        let bass_in_buf_size = treble_in_buf_size * 8;

        if bars_per_channel > treble_in_buf_size as usize / 2 + 1 {
            return Err(Error::TooHighAmountBars {
                amount_bars: bars_per_channel as u32,
                sample_rate,
                max_amount_bars: (treble_in_buf_size / 2 + 1) as u32,
            });
        }

        if freq_range.is_empty() {
            return Err(Error::EmptyFreqRange {
                start: freq_range.start,
                end: freq_range.end,
            });
        }

        if freq_range.end as u32 > sample_rate / 2 {
            return Err(Error::NyquistIssue {
                freq: freq_range.end,
                max_freq: sample_rate / 2,
            });
        }

        let left = AudioData::new(treble_in_buf_size);
        let right = AudioData::new(treble_in_buf_size);

        let input_buffer_size = bass_in_buf_size * audio_channels as usize;

        let bass_multiplier: Vec<f64> = (0..bass_in_buf_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (bass_in_buf_size as f64 - 1.0)).cos()))
            .collect();
        let mid_multiplier: Vec<f64> = (0..mid_in_buf_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (mid_in_buf_size as f64 - 1.0)).cos()))
            .collect();
        let treble_multiplier: Vec<f64> = (0..treble_in_buf_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (treble_in_buf_size as f64 - 1.0)).cos()))
            .collect();

        let total_amount_bars = bars_per_channel * audio_channels as usize;
        let prev_cava_out = vec![0.0; total_amount_bars];
        let cava_mem = vec![0.0; total_amount_bars];
        let input_buffer = vec![0.0; input_buffer_size as usize];
        let cava_peak = vec![0.0; total_amount_bars];

        let eq = vec![0.0; bars_per_channel + 1];
        let cut_off_frequency = vec![0.0; bars_per_channel + 1];
        let fft_buffer_lower_cut_off = vec![0; bars_per_channel + 1];
        let fft_buffer_upper_cut_off = vec![0; bars_per_channel + 1];
        let cava_fall = vec![0.0; total_amount_bars];
        let cava_out = vec![0.0; total_amount_bars];

        let mut cava = Cava {
            bars_per_channel,
            audio_channels,
            sample_rate,
            bass_cut_off_bar: -1,
            treble_cut_off_bar: -1,
            init_sens: true,
            enable_autosens,
            frame_skip: 1,

            sens: 1.0,
            framerate: 75.0,
            noise_reduction,

            left,
            right,

            bass_multiplier,
            mid_multiplier,
            treble_multiplier,

            prev_cava_out,
            cava_mem,
            input_buffer,
            cava_peak,
            cava_out,

            eq,
            cut_off_frequency,
            fft_buffer_lower_cut_off,
            fft_buffer_upper_cut_off,
            cava_fall,
        };

        // Calculate cutoff frequencies and EQ
        let lower_cut_off = freq_range.start as f64;
        let upper_cut_off = freq_range.end as f64;
        let bass_cut_off = 100.0;
        let treble_cut_off = 500.0;

        let frequency_constant =
            (lower_cut_off / upper_cut_off).log10() / (1.0 / (bars_per_channel as f64 + 1.0) - 1.0);

        let mut relative_cut_off = vec![0.0; bars_per_channel + 1];

        let mut first_bar = true;
        #[allow(unused_assignments)]
        let mut first_treble_bar = false;
        let mut bar_buffer = vec![0; bars_per_channel + 1];

        for n in 0..=bars_per_channel {
            let bar_distribution_coefficient = frequency_constant * -1.0
                + ((n as f64 + 1.0) / (bars_per_channel as f64 + 1.0)) * frequency_constant;
            cava.cut_off_frequency[n] = upper_cut_off * 10.0f64.powf(bar_distribution_coefficient);

            if n > 0 {
                if cava.cut_off_frequency[n - 1] >= cava.cut_off_frequency[n]
                    && cava.cut_off_frequency[n - 1] > bass_cut_off
                {
                    cava.cut_off_frequency[n] = cava.cut_off_frequency[n - 1]
                        + (cava.cut_off_frequency[n - 1] - cava.cut_off_frequency[n - 2]);
                }
            }

            relative_cut_off[n] = cava.cut_off_frequency[n] / (cava.sample_rate as f64 / 2.0);

            cava.eq[n] = cava.cut_off_frequency[n].powf(1.0);
            cava.eq[n] /= 2.0f64.powf(29.0);
            cava.eq[n] /= (bass_in_buf_size as f64).log2();

            if cava.cut_off_frequency[n] < bass_cut_off {
                bar_buffer[n] = 1;
                cava.fft_buffer_lower_cut_off[n] =
                    (relative_cut_off[n] * (bass_in_buf_size as f64 / 2.0)) as usize;
                cava.bass_cut_off_bar += 1;
                cava.treble_cut_off_bar += 1;
                if cava.bass_cut_off_bar > 0 {
                    first_bar = false;
                }

                if cava.fft_buffer_lower_cut_off[n] > bass_in_buf_size / 2 {
                    cava.fft_buffer_lower_cut_off[n] = bass_in_buf_size / 2;
                }
            } else if cava.cut_off_frequency[n] > bass_cut_off
                && cava.cut_off_frequency[n] < treble_cut_off
            {
                bar_buffer[n] = 2;
                cava.fft_buffer_lower_cut_off[n] =
                    (relative_cut_off[n] * (mid_in_buf_size as f64 / 2.0)) as usize;
                cava.treble_cut_off_bar += 1;
                if (cava.treble_cut_off_bar - cava.bass_cut_off_bar) == 1 {
                    first_bar = true;
                    if n > 0 {
                        cava.fft_buffer_upper_cut_off[n - 1] =
                            (relative_cut_off[n] * (bass_in_buf_size as f64 / 2.0)) as usize;
                    }
                } else {
                    first_bar = false;
                }

                if cava.fft_buffer_lower_cut_off[n] > mid_in_buf_size / 2 {
                    cava.fft_buffer_lower_cut_off[n] = mid_in_buf_size / 2;
                }
            } else {
                bar_buffer[n] = 3;
                cava.fft_buffer_lower_cut_off[n] =
                    (relative_cut_off[n] * (treble_in_buf_size as f64 / 2.0)) as usize;
                first_treble_bar = true;
                if first_treble_bar {
                    first_bar = true;
                    if n > 0 {
                        cava.fft_buffer_upper_cut_off[n - 1] =
                            (relative_cut_off[n] * (mid_in_buf_size as f64 / 2.0)) as usize;
                    }
                } else {
                    first_bar = false;
                }

                if cava.fft_buffer_lower_cut_off[n] > treble_in_buf_size / 2 {
                    cava.fft_buffer_lower_cut_off[n] = treble_in_buf_size / 2;
                }
            }

            if n > 0 {
                if !first_bar {
                    cava.fft_buffer_upper_cut_off[n - 1] = cava.fft_buffer_lower_cut_off[n] - 1;

                    if cava.fft_buffer_lower_cut_off[n] <= cava.fft_buffer_lower_cut_off[n - 1] {
                        let room_for_more = match bar_buffer[n] {
                            1 => {
                                cava.fft_buffer_lower_cut_off[n - 1] + 1 < bass_in_buf_size / 2 + 1
                            }
                            2 => cava.fft_buffer_lower_cut_off[n - 1] + 1 < mid_in_buf_size / 2 + 1,
                            3 => {
                                cava.fft_buffer_lower_cut_off[n - 1] + 1
                                    < treble_in_buf_size / 2 + 1
                            }
                            _ => false,
                        };

                        if room_for_more {
                            cava.fft_buffer_lower_cut_off[n] =
                                cava.fft_buffer_lower_cut_off[n - 1] + 1;
                            cava.fft_buffer_upper_cut_off[n - 1] =
                                cava.fft_buffer_lower_cut_off[n] - 1;

                            relative_cut_off[n] = match bar_buffer[n] {
                                1 => {
                                    cava.fft_buffer_lower_cut_off[n] as f64
                                        / (bass_in_buf_size as f64 / 2.0)
                                }
                                2 => {
                                    cava.fft_buffer_lower_cut_off[n] as f64
                                        / (mid_in_buf_size as f64 / 2.0)
                                }
                                3 => {
                                    cava.fft_buffer_lower_cut_off[n] as f64
                                        / (treble_in_buf_size as f64 / 2.0)
                                }
                                _ => 0.0,
                            };

                            cava.cut_off_frequency[n] =
                                relative_cut_off[n] * (cava.sample_rate as f64 / 2.0);
                        }
                    }
                } else {
                    if cava.fft_buffer_upper_cut_off[n - 1] <= cava.fft_buffer_lower_cut_off[n - 1]
                    {
                        cava.fft_buffer_upper_cut_off[n - 1] =
                            cava.fft_buffer_lower_cut_off[n - 1] + 1;
                    }
                }
            }
        }

        Ok(cava)
    }

    pub fn execute(&mut self, new_samples: &[f64]) -> &[f64] {
        let new_samples_len = new_samples.len();
        let input_buffer_len = self.input_buffer.len();

        if new_samples_len > input_buffer_len {
            return &self.cava_out;
        }

        let silence = new_samples.iter().all(|&v| v == 0.0);
        if !new_samples.is_empty() {
            self.framerate -= self.framerate / 64.0;
            self.framerate += ((self.sample_rate as f64
                * self.audio_channels as usize as f64
                * self.frame_skip as f64)
                / new_samples.len() as f64)
                / 64.0;
            self.frame_skip = 1;

            self.input_buffer
                .copy_within(..input_buffer_len - new_samples_len, new_samples_len);
            self.input_buffer[..new_samples_len].copy_from_slice(new_samples);
        } else {
            self.frame_skip += 1;
        }

        for n in 0..self.left.in_bass.len() {
            match self.audio_channels {
                Channels::Mono => {
                    self.left.in_bass[n] = self.bass_multiplier[n] * self.input_buffer[n];
                }
                Channels::Stereo => {
                    self.left.in_bass[n] = self.bass_multiplier[n] * self.input_buffer[n * 2 + 1];
                    self.right.in_bass[n] = self.bass_multiplier[n] * self.input_buffer[n * 2];
                }
            }
        }

        for n in 0..self.left.in_mid.len() {
            match self.audio_channels {
                Channels::Mono => {
                    self.left.in_mid[n] = self.mid_multiplier[n] * self.input_buffer[n];
                }
                Channels::Stereo => {
                    self.left.in_mid[n] = self.mid_multiplier[n] * self.input_buffer[n * 2 + 1];
                    self.right.in_mid[n] = self.mid_multiplier[n] * self.input_buffer[n * 2];
                }
            }
        }

        for n in 0..self.left.in_treble.len() {
            match self.audio_channels {
                Channels::Mono => {
                    self.left.in_treble[n] = self.treble_multiplier[n] * self.input_buffer[n];
                }
                Channels::Stereo => {
                    self.left.in_treble[n] =
                        self.treble_multiplier[n] * self.input_buffer[n * 2 + 1];
                    self.right.in_treble[n] = self.treble_multiplier[n] * self.input_buffer[n * 2];
                }
            }
        }

        self.left
            .bass
            .process(&mut self.left.in_bass, &mut self.left.out_bass)
            .unwrap();
        self.left
            .mid
            .process(&mut self.left.in_mid, &mut self.left.out_mid)
            .unwrap();
        self.left
            .treble
            .process(&mut self.left.in_treble, &mut self.left.out_treble)
            .unwrap();

        if self.audio_channels == Channels::Stereo {
            self.right
                .bass
                .process(&mut self.right.in_bass, &mut self.right.out_bass)
                .unwrap();
            self.right
                .mid
                .process(&mut self.right.in_mid, &mut self.right.out_mid)
                .unwrap();
            self.right
                .treble
                .process(&mut self.right.in_treble, &mut self.right.out_treble)
                .unwrap();
        }

        for n in 0..self.bars_per_channel {
            let mut temp_l = 0.0;
            let mut temp_r = 0.0;

            for i in self.fft_buffer_lower_cut_off[n]..=self.fft_buffer_upper_cut_off[n] {
                if n <= self.bass_cut_off_bar as usize {
                    temp_l += self.left.out_bass[i].norm();
                    if self.audio_channels == Channels::Stereo {
                        temp_r += self.right.out_bass[i].norm();
                    }
                } else if n > self.bass_cut_off_bar as usize
                    && n <= self.treble_cut_off_bar as usize
                {
                    temp_l += self.left.out_mid[i].norm();
                    if self.audio_channels == Channels::Stereo {
                        temp_r += self.right.out_mid[i].norm();
                    }
                } else if n > self.treble_cut_off_bar as usize {
                    temp_l += self.left.out_treble[i].norm();
                    if self.audio_channels == Channels::Stereo {
                        temp_r += self.right.out_treble[i].norm();
                    }
                }
            }

            temp_l /=
                (self.fft_buffer_upper_cut_off[n] - self.fft_buffer_lower_cut_off[n] + 1) as f64;
            temp_l *= self.eq[n];
            self.cava_out[n] = temp_l;

            if self.audio_channels == Channels::Stereo {
                temp_r /= (self.fft_buffer_upper_cut_off[n] - self.fft_buffer_lower_cut_off[n] + 1)
                    as f64;
                temp_r *= self.eq[n];
                self.cava_out[n + self.bars_per_channel] = temp_r;
            }
        }

        if self.enable_autosens {
            for n in 0..self.bars_per_channel * self.audio_channels as usize {
                self.cava_out[n] *= self.sens;
            }
        }

        let gravity_mod = (60.0 / self.framerate).powf(2.5) * 1.54 / self.noise_reduction;
        let gravity_mod = if gravity_mod < 1.0 { 1.0 } else { gravity_mod };

        let mut overshoot = false;

        for n in 0..self.bars_per_channel * self.audio_channels as usize {
            if self.cava_out[n] < self.prev_cava_out[n] && self.noise_reduction > 0.1 {
                self.cava_out[n] = self.cava_peak[n]
                    * (1.0 - (self.cava_fall[n] * self.cava_fall[n] * gravity_mod));
                if self.cava_out[n] < 0.0 {
                    self.cava_out[n] = 0.0;
                }
                self.cava_fall[n] += 0.028;
            } else {
                self.cava_peak[n] = self.cava_out[n];
                self.cava_fall[n] = 0.0;
            }
            self.prev_cava_out[n] = self.cava_out[n];

            self.cava_out[n] = self.cava_mem[n] * self.noise_reduction + self.cava_out[n];
            self.cava_mem[n] = self.cava_out[n];

            if self.enable_autosens && self.cava_out[n] > 1.0 {
                overshoot = true;
            }
        }

        if self.enable_autosens {
            if overshoot {
                self.sens *= 0.98;
                self.init_sens = false;
            } else if !silence {
                self.sens *= 1.002;
                if self.init_sens {
                    self.sens *= 1.1;
                }
            }
        }

        &self.cava_out
    }
}

struct AudioData<F: FftNum> {
    bass: Arc<dyn RealToComplex<F>>,
    mid: Arc<dyn RealToComplex<F>>,
    treble: Arc<dyn RealToComplex<F>>,

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
            bass,
            mid,
            treble,

            out_bass,
            out_mid,
            out_treble,

            in_bass,
            in_mid,
            in_treble,
        }
    }
}
