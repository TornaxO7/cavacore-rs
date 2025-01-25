use std::ops::Range;

use crate::Cava;

pub type Hz = u32;

#[repr(u8)]
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
    pub fn build(&self) -> Cava {
        let plan = unsafe {
            bindings::cava_init(
                number_of_bars as i32,
                sample_rate,
                channel as std::os::raw::c_int,
                opts.enable_autosens as i32,
                opts.noise_reduction,
                opts.freq_range.start as i32,
                opts.freq_range.end as i32,
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
        } else {
            unreachable!("Invalid cava status was set: {}", unsafe { *plan }.status);
        }

        Ok(Cava { plan })
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
