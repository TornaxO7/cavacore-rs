#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("{amount_bars} bars for a sample rate of {sample_rate} can't be more than {max_amount_bars} bars")]
    TooHighAmountBars {
        amount_bars: usize,
        sample_rate: u32,
        max_amount_bars: usize,
    },

    #[error("Noise reduction has to be within the range [0..1]. Your value: {0}.")]
    InvalidNoiseReduction(f64),

    #[error("Frequency start and end frequency musn't be equal but given frequency range: {start}..{end} (Hz)")]
    EmptyFreqRange { start: u32, end: u32 },

    #[error("Due to the Nyquist sampling theorem the highesest frequency cutoff does not exceed 'sample rate' / 2 (= {max_freq}) but you set it to {freq}")]
    NyquistIssue { freq: u32, max_freq: u32 },
}
