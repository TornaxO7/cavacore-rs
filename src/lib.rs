use error::Error;

mod bindings;
mod error;
mod wrapper;

use std::ops::Range;

pub use wrapper::Cava;

pub type Hz = u32;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Channel {
    Mono,
    Stereo,
}

#[derive(Debug, Clone)]
pub struct Builder {
    pub amount_bars: u16,
    pub sample_rate: u32,
    pub channel: Channel,
    pub enable_autosens: bool,
    pub noise_reduction: f64,
    pub freq_range: Range<Hz>,
}

impl Builder {
    pub fn build(&self) -> Result<Cava, Error> {
        let plan = unsafe {
            bindings::cava_init(
                self.amount_bars as i32,
                self.sample_rate,
                self.channel as std::os::raw::c_int,
                self.enable_autosens as i32,
                self.noise_reduction,
                self.freq_range.start as i32,
                self.freq_range.end as i32,
            )
        };

        // sanity checks
        if plan.is_null() {
            return Err(Error::AllocCava);
        } else if unsafe { *plan }.status == -1 {
            let plan = unsafe { *plan };

            let err_msg_bytes = plan
                .error_message
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<u8>>();

            let err_msg = String::from_utf8(err_msg_bytes).unwrap();

            return Err(Error::Init(err_msg));
        }

        Ok(unsafe { Cava::new(plan) })
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            amount_bars: 32,
            sample_rate: 44_200,
            channel: Channel::Mono,
            enable_autosens: true,
            noise_reduction: 0.77,
            freq_range: 50..10_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[test]
    fn cavacore_test() {
        let bars_per_channel = 10;
        let channel = crate::Channel::Stereo;
        let buffer_size = 512 * channel as usize;
        let sample_rate = 44_100;

        let builder = Builder {
            amount_bars: bars_per_channel,
            sample_rate,
            channel: Channel::Stereo,
            enable_autosens: true,
            noise_reduction: 0.77,
            freq_range: 50..10_000,
        };

        let mut cava = builder.build().unwrap();
        let mut cava_in =
            vec![0.; bars_per_channel as usize * builder.channel as usize].into_boxed_slice();

        // running cava execute 300 times (simulating about 3.5 seconds run time)
        for k in 0..300 {
            for n in 0..buffer_size / 2 {
                cava_in[n * 2] = (2. * f64::consts::PI * 200. / sample_rate as f64
                    * (n as f64 + (k as f64 * buffer_size as f64 / 2.)))
                    .sin()
                    * 20_000.;

                cava_in[n * 2 + 1] = cava_in[n * 2];
            }

            cava.execute(&mut cava_in);
        }
    }
}
