use std::num::NonZeroUsize;

use crate::{Cava, Error, SampleRate};

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

        self.bars_per_channel = bars_per_channel;
        self.update_state();
        Ok(())
    }

    pub fn set_sample_rate(&mut self, sample_rate: SampleRate) -> Result<(), Vec<Error>> {
        let sample_rate = u32::from(sample_rate);
        let mut errors = Vec::new();

        let treble_buffer_size = Self::compute_treble_buffer_size(sample_rate);
        if self.bars_per_channel > treble_buffer_size / 2 + 1 {
            errors.push(Error::TooHighAmountBars {
                amount_bars: self.bars_per_channel,
                sample_rate,
                max_amount_bars: treble_buffer_size / 2 + 1,
            });
        }

        if self.freq_range.end > sample_rate / 2 {
            errors.push(Error::NyquistIssue {
                freq: self.freq_range.end,
                max_freq: sample_rate / 2,
            });
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        self.sample_rate = sample_rate;
        self.update_state();
        Ok(())
    }
}
