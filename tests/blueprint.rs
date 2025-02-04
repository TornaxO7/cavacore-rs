use std::{
    f64::consts,
    num::{NonZeroU32, NonZeroUsize},
};

use cavacore::{Cava, CavaOpts, SampleRate};

const SAMPLE_RATE: u32 = 44_100;
const BARS_PER_CHANNEL: NonZeroUsize = NonZeroUsize::new(10).unwrap();

#[test]
fn blueprint() {
    let mut cava = Cava::new(CavaOpts {
        bars_per_channel: BARS_PER_CHANNEL,
        audio_channels: cavacore::Channels::Stereo,
        sample_rate: SampleRate::new(SAMPLE_RATE).unwrap(),
        noise_reduction: 0.77,
        frequency_range: NonZeroU32::new(50).unwrap()..NonZeroU32::new(10_000).unwrap(),
        ..Default::default()
    })
    .unwrap();

    test_cava(&mut cava);
    println!("Default cava works.");

    // after changing values, nothing should change
    cava.set_bars(BARS_PER_CHANNEL).unwrap();
    test_cava(&mut cava);
    println!("After `set_bars` works.");

    cava.set_sample_rate(SampleRate::new(SAMPLE_RATE).unwrap())
        .unwrap();
    test_cava(&mut cava);
    println!("After `set_sample_rate` works.");
}

fn test_cava(cava: &mut Cava) {
    let blueprint_2000_mhz = [0., 0., 0., 0., 0., 0., 0.493, 0.446, 0., 0.];
    let blueprint_200_mhz = [0., 0., 0.978, 0.008, 0., 0.001, 0., 0., 0., 0.];

    const BUFFER_SIZE: usize = 512 * 2;
    let mut input = vec![0.; BUFFER_SIZE];
    let mut output = cava.make_output();
    for k in 0..300 {
        for n in 0..BUFFER_SIZE / 2 {
            input[n * 2] = (2. * consts::PI * 200. / SAMPLE_RATE as f64
                * (n + (k * BUFFER_SIZE / 2)) as f64)
                * 20_000.;
            input[n * 2 + 1] = (2. * consts::PI * 2_000. / SAMPLE_RATE as f64
                * (n + (k * BUFFER_SIZE / 2)) as f64)
                * 20_000.;
        }

        cava.execute(&input, &mut output);
    }

    // rounding last output to nearst 1/1000th
    for val in output.iter_mut() {
        *val = (*val * 1000.).round() / 1000.;
    }

    // checking if within 2% of blueprint
    for i in 0..usize::from(BARS_PER_CHANNEL) {
        let out = output[i];
        let expected = blueprint_200_mhz[i];

        assert!(
            out > expected * 1.02 || out < expected * 0.98,
            "{0}: {1} > {2} * 1.02 || {1} < {2} * 0.98",
            i,
            out,
            expected
        );
    }

    for i in 0..usize::from(BARS_PER_CHANNEL) {
        let out = output[i + usize::from(BARS_PER_CHANNEL)];
        let expected = blueprint_2000_mhz[i];

        assert!(
            out > expected * 1.02 || out < expected * 0.98,
            "{0}: {1} > {2} * 1.02 || {1} < {2} * 0.98",
            i,
            out,
            expected
        );
    }
}
