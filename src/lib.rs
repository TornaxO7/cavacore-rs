//! A safe rust wrapper of the [cavacore](https://github.com/karlstav/cava/blob/master/CAVACORE.md) engine.
//!
//! # Example
//! ```rust
//! use cava_rs::{Builder, Cava, Channel};
//!
//! // Configure cava with the builder first...
//! let builder = Builder {
//!     // we will only listen to one channel
//!     channel: Channel::Mono,
//!     .. Builder::default()
//! };
//!
//! let mut cava = builder.build().expect("Build cava");
//!
//! // feed cava with some samples
//! let mut new_samples: [f64; 3] = [1., 2., 3.];
//!
//! // and let it give you the bars back
//! let bars = cava.execute(&mut new_samples);
//! ```
use error::Error;

mod bindings;
mod error;
mod wrapper;

use std::ops::Range;

pub use wrapper::Cava;

/// Type alias for better readability.
pub type Hz = u32;

/// Sets the amount of channels.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Channel {
    /// Represents only one channel.
    Mono = 1,

    /// Represents two channels.
    Stereo,
}

/// Main struct to create a new instance of cava.
///
/// # Example
/// ```rust
/// use cava_rs::{Builder, Cava, Channel};
///
/// let builder = Builder {
///     bars_per_channel: 10,
///     channel: Channel::Mono,
///     .. Builder::default()
/// };
///
/// let cava = builder.build().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Builder {
    /// Amount of bars per channel.
    pub bars_per_channel: u16,

    /// The sample rate of the input signal.
    pub sample_rate: u32,

    /// The number of interleaved channels of the input.
    pub channel: Channel,

    /// Toggle automatic sensitivity adjustment.
    ///
    /// If `true`: Gives a dynamically adjusted output signal from 0 to 1
    ///            the output is continously adjusted to use the entire range.
    /// If `false`: Will pass the raw values from cava directly to the output
    ///             the max values will then be dependent on the input.
    pub enable_autosens: bool,

    /// Adjust noise reduciton filters. **Has** to be within the range 0..1.
    /// Its default value contains the recommended value.
    ///
    /// The raw visualization is very noisy, this factor adjusts the integral
    /// and gravity filters inside cavacore to keep the signal smooth.
    /// 1 will be very slow and smooth, 0 will be fast but noisy.
    pub noise_reduction: f64,

    /// The frequency range which cava will use for the visualisation.
    pub freq_range: Range<Hz>,
}

impl Builder {
    /// Create a new cava instance with the current settings.
    pub fn build(&self) -> Result<Cava, Error> {
        if !(0. <= self.noise_reduction && self.noise_reduction <= 1.) {
            let err_msg = format!(
                "`noise_reduction` has to be within the range `0..1`. Current value: {}",
                self.noise_reduction
            );
            return Err(Error::Init(err_msg));
        }

        let plan = unsafe {
            bindings::cava_init(
                self.bars_per_channel as i32,
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
            bars_per_channel: 32,
            sample_rate: 44_100,
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
            bars_per_channel,
            sample_rate,
            channel: Channel::Stereo,
            enable_autosens: true,
            noise_reduction: 0.77,
            freq_range: 50..10_000,
        };

        let mut cava = builder.build().unwrap();
        let mut cava_in = vec![0.; buffer_size as usize].into_boxed_slice();

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
