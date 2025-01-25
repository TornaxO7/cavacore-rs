use crate::bindings;

#[derive(Debug)]
pub struct Cava {
    plan: *mut bindings::cava_plan,
}

impl Cava {
    pub(crate) fn new(plan: *mut bindings::cava_plan) -> Self {
        Self { plan }
    }
}

impl Drop for Cava {
    fn drop(&mut self) {
        unsafe {
            bindings::cava_destroy(self.plan);
        }
    }
}
