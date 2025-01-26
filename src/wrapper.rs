use crate::bindings;

/// The wrapper of `cavacore`.
/// Can be created by using [`Builder`].
#[derive(Debug)]
pub struct Cavacore {
    plan: *mut bindings::cava_plan,

    out_buffer: Box<[f64]>,
}

impl Cavacore {
    // SAFETY: `plan` must point to a valid cava plan
    pub(crate) unsafe fn new(plan: *mut bindings::cava_plan) -> Self {
        debug_assert!(!plan.is_null());

        let plan_deref = unsafe { *plan };
        debug_assert!(plan_deref.status == 0);

        let out_buffer = vec![0.; (plan_deref.number_of_bars * plan_deref.audio_channels) as usize]
            .into_boxed_slice();

        Self { plan, out_buffer }
    }

    /// Execute the visualisation by providing some new samples of the source audio and
    /// return the values of the bars back.
    ///
    /// # Example
    /// ```rust
    /// use cavacore::{Builder, Cavacore};
    ///
    /// let builder = Builder::default();
    ///
    /// let mut cava = builder.build().unwrap();
    ///
    /// // there can be also no new samples coming from the source
    /// cava.execute(&mut []);
    /// ```
    pub fn execute(&mut self, new_samples: &mut [f64]) -> &[f64] {
        unsafe {
            bindings::cava_execute(
                new_samples.as_mut_ptr(),
                new_samples.len() as i32,
                self.out_buffer.as_mut_ptr(),
                self.plan,
            );
        }

        &self.out_buffer
    }
}

impl Drop for Cavacore {
    fn drop(&mut self) {
        unsafe {
            bindings::cava_destroy(self.plan);
        }
    }
}
