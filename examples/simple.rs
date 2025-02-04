use std::num::NonZeroUsize;

use cavacore::{Cava, CavaOpts, SampleRate};

fn main() {
    let mut cava = Cava::new(CavaOpts {
        // set some values when initialising cava
        sample_rate: SampleRate::new(44_100).unwrap(),
        ..Default::default()
    })
    .unwrap();

    // feed cava with some samples
    let new_samples = [1., 2., 3.];
    let mut bars = cava.make_output();

    cava.execute(&new_samples, &mut bars);

    // you can also change some values afterwards
    cava.set_bars(NonZeroUsize::new(10).unwrap()).unwrap();
}
