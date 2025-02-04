use std::num::NonZeroUsize;

use crate::{Cava, Error};

impl Cava {
    pub fn set_bars(&mut self, bars_per_channel: NonZeroUsize) -> Result<(), Error> {
        let bars_per_channel = usize::from(bars_per_channel);

        // validation check
        let treble_buffer_size = Self::compute_treble_buffer_size(self.sample_rate);
        if bars_per_channel > treble_buffer_size / 2 + 1 {
            return Err(Error::TooHighAmountBars {
                amount_bars: bars_per_channel,
                sample_rate: self.sample_rate,
                max_amount_bars: treble_buffer_size / 2 + 1,
            });
        }

        self.buffer_lower_cut_off.resize(bars_per_channel + 1, 0);
        self.buffer_upper_cut_off.resize(bars_per_channel + 1, 0);
        self.eq.resize(bars_per_channel + 1, 0.);

        // process: calculate cutoff frequencies and eq
        let bass_cut_off = 100.;
        let treble_cut_off = 500.;
        let mut cut_off_frequency = vec![0f64; bars_per_channel + 1];

        // calculate frequency constant (used to distribute bars across the frequency band)
        let frequency_constant = (self.freq_range.start as f64 / self.freq_range.end as f64)
            .log10()
            / (1. / (bars_per_channel as f64 + 1.) - 1.);

        let mut relative_cut_off = vec![0.; bars_per_channel + 1].into_boxed_slice();
        let mut bass_cut_off_bar = -1;
        let mut treble_cut_off_bar = -1;
        let mut first_bar = true;
        let mut first_treble_bar = 0;
        let mut bar_buffer = vec![0; bars_per_channel + 1];

        for n in 0..bars_per_channel + 1 {
            let mut bar_distribution_coefficient = -frequency_constant;
            bar_distribution_coefficient +=
                (n as f64 + 1.) / (bars_per_channel as f64 + 1.) * frequency_constant;

            cut_off_frequency[n] =
                self.freq_range.end as f64 * 10f64.powf(bar_distribution_coefficient);

            if n > 0 {
                // what?
                let condition = cut_off_frequency[n - 1] >= cut_off_frequency[n]
                    && cut_off_frequency[n - 1] > bass_cut_off;

                if condition {
                    cut_off_frequency[n] = cut_off_frequency[n - 1]
                        + (cut_off_frequency[n - 1] - cut_off_frequency[n - 2]);
                }
            }

            relative_cut_off[n] = cut_off_frequency[n] / (self.sample_rate as f64 / 2.);

            // some random magic?
            self.eq[n] = cut_off_frequency[n].powf(1.);
            self.eq[n] /= 2f64.powf(29.);
            self.eq[n] /= (self.left.in_bass.len() as f64).log2();

            if cut_off_frequency[n] < bass_cut_off {
                // BASS
                bar_buffer[n] = 1;
                self.buffer_lower_cut_off[n] =
                    relative_cut_off[n] as u32 * (self.left.in_bass.len() as u32 / 2);
                bass_cut_off_bar += 1;
                treble_cut_off_bar += 1;
                if bass_cut_off_bar > 0 {
                    first_bar = false;
                }

                if self.buffer_lower_cut_off[n] > self.left.in_bass.len() as u32 / 2 {
                    self.buffer_lower_cut_off[n] = self.left.in_bass.len() as u32 / 2;
                }
            } else if cut_off_frequency[n] > bass_cut_off && cut_off_frequency[n] < treble_cut_off {
                // MID
                bar_buffer[n] = 2;
                self.buffer_lower_cut_off[n] =
                    relative_cut_off[n] as u32 * (self.left.in_mid.len() as u32 / 2);
                treble_cut_off_bar += 1;

                if (treble_cut_off_bar - bass_cut_off_bar) == 1 {
                    first_bar = true;
                    if n > 0 {
                        self.buffer_upper_cut_off[n - 1] =
                            relative_cut_off[n] as u32 * (self.left.in_bass.len() as u32 / 2);
                    }
                } else {
                    first_bar = false;
                }

                if self.buffer_lower_cut_off[n] > self.left.in_mid.len() as u32 / 2 {
                    self.buffer_lower_cut_off[n] = self.left.in_mid.len() as u32 / 2;
                }
            } else {
                // TREBLE
                bar_buffer[n] = 3;
                self.buffer_lower_cut_off[n] =
                    relative_cut_off[n] as u32 * (self.left.in_treble.len() as u32 / 2);
                first_treble_bar += 1;
                if first_treble_bar == 1 {
                    first_bar = true;
                    if n > 0 {
                        self.buffer_upper_cut_off[n - 1] =
                            relative_cut_off[n] as u32 * (self.left.in_mid.len() as u32 / 2);
                    }
                } else {
                    first_bar = false;
                }

                if self.buffer_lower_cut_off[n] > self.left.in_treble.len() as u32 / 2 {
                    self.buffer_lower_cut_off[n] = self.left.in_treble.len() as u32 / 2;
                }
            }

            if n > 0 {
                if !first_bar {
                    self.buffer_upper_cut_off[n - 1] =
                        self.buffer_lower_cut_off[n].saturating_sub(1);

                    if self.buffer_lower_cut_off[n] <= self.buffer_lower_cut_off[n - 1] {
                        let mut room_for_more = false;

                        if bar_buffer[n] == 1 {
                            if self.buffer_lower_cut_off[n - 1] + 1
                                < self.left.in_bass.len() as u32 / 2 + 1
                            {
                                room_for_more = true;
                            }
                        } else if bar_buffer[n] == 2 {
                            if self.buffer_lower_cut_off[n - 1] + 1
                                < self.left.in_mid.len() as u32 / 2 + 1
                            {
                                room_for_more = true;
                            }
                        } else if bar_buffer[n] == 3
                            && self.buffer_lower_cut_off[n - 1] + 1
                                < self.left.in_treble.len() as u32 / 2 + 1
                        {
                            room_for_more = true;
                        }

                        if room_for_more {
                            // push the spectrum up
                            self.buffer_lower_cut_off[n] = self.buffer_lower_cut_off[n - 1] + 1;
                            self.buffer_upper_cut_off[n - 1] = self.buffer_lower_cut_off[n] - 1;

                            // calculate new cut off frequency
                            if bar_buffer[n] == 1 {
                                relative_cut_off[n] = self.buffer_lower_cut_off[n] as f64
                                    / (self.left.in_bass.len() as f64 / 2.);
                            } else if bar_buffer[n] == 2 {
                                relative_cut_off[n] = self.buffer_lower_cut_off[n] as f64
                                    / (self.left.in_mid.len() as f64 / 2.);
                            } else if bar_buffer[n] == 3 {
                                relative_cut_off[n] = self.buffer_lower_cut_off[n] as f64
                                    / (self.left.in_treble.len() as f64 / 2.);
                            }

                            cut_off_frequency[n] =
                                relative_cut_off[n] * (self.sample_rate as f64 / 2.);
                        }
                    }
                } else if self.buffer_upper_cut_off[n - 1] <= self.buffer_lower_cut_off[n - 1] {
                    self.buffer_upper_cut_off[n - 1] = self.buffer_lower_cut_off[n - 1] + 1;
                }
            }
        }

        Ok(())
    }
}
