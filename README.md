# Cavacore-rs

A rewrite in rust of [cavacore].

# Example

```rs
use cavacore::{CavaBuilder, Cava, Channels};

let mut cava = CavaBuilder::default()
    .audio_channels(Channels::Mono)
    .build()
    .unwrap();

// feed cava with some samples
let mut new_samples: [f64; 3] = [1., 2., 3.];
let mut bars = cava.make_output();

// and let it give you the bars
cava.execute(&new_samples, &mut bars);
```

[cavacore]: https://github.com/karlstav/cava/blob/master/CAVACORE.md
