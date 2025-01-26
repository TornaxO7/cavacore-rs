#[derive(Debug, Clone)]
pub enum Error {
    /// Couldn't allocate buffers for cava.
    AllocCava,
    /// An error occured during initialization.
    Init(String),
}
