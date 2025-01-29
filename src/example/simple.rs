fn main() {
    let mut cava = CavaBuilder::default()
        .bars_per_channel(NonZeroUsize::new(10).unwrap())
        .build();

    let bars = cava.execute(&[1., 2., 3.]);
}
