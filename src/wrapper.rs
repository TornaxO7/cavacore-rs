use crate::bindings;

#[derive(Debug)]
pub struct Cava {
    plan: *mut bindings::cava_plan,

    out_buffer: Box<[f64]>,
}

impl Cava {
    /// Safety: `plan` must point to a valid cava plan
    pub(crate) unsafe fn new(plan: *mut bindings::cava_plan) -> Self {
        debug_assert!(!plan.is_null());

        let plan_deref = unsafe { *plan };
        debug_assert!(plan_deref.status == 0);

        let out_buffer = vec![0.; (plan_deref.number_of_bars * plan_deref.audio_channels) as usize]
            .into_boxed_slice();

        Self { plan, out_buffer }
    }

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

impl Drop for Cava {
    fn drop(&mut self) {
        unsafe {
            bindings::cava_destroy(self.plan);
        }
    }
}
