#[derive(Debug, Clone)]
pub enum Error {
    AllocCava,
    /// An error occured during initialization.
    Init(String),
}
