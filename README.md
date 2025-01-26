# Cava-rs

A rust wrapper for [cavacore].

# Required dependencies

- [`fftw3`]

# Example

```rs
use cavacore::{Builder, Cavacore, Channel};

// Configure cava with the builder first...
let builder = Builder {
    // we will only listen to one channel
    channel: Channel::Mono,
    .. Builder::default()
};

let mut cava = builder.build().expect("Build cava");

// feed cava with some samples
let mut new_samples: [f64; 3] = [1., 2., 3.];

// and let it give you the bars back
let bars = cava.execute(&mut new_samples);
```

[cavacore]: https://github.com/karlstav/cava/blob/master/CAVACORE.md
[fftw]: http://www.fftw.org/
