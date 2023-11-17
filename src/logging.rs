use std::io::IsTerminal;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub fn setup() {
    let reg = tracing_subscriber::registry().with(EnvFilter::from_default_env());
    let log_fmt = std::env::var("LOG_FMT").unwrap_or_default().to_lowercase();
    match (std::io::stdout().is_terminal(), log_fmt.as_str()) {
        (false, _) | (_, "json") => reg.with(fmt::layer().json()).init(),
        _ => reg.with(fmt::layer().pretty()).init(),
    };
}

pub mod prelude {
    pub use tracing::{debug, error, info, trace, warn};
}
