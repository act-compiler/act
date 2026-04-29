use std::{
    collections::HashSet,
    sync::{LazyLock, Mutex},
    time::Duration,
};

use egg::{Id, Subst, Symbol};

pub static PROCESSED: LazyLock<Mutex<HashSet<(Symbol, Id, Subst)>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

// Log file handle (None = discard verbose output)
pub static LOG_FILE: LazyLock<Mutex<Option<std::fs::File>>> =
    LazyLock::new(|| Mutex::new(None));

// External Timeout
pub const TIME_LIMIT: Duration = Duration::from_millis(1000);
// Number of iterations after which the extractor runs
pub const N: usize = 1;
// Max PiiGraph size for Slow Extractor Algorithm
pub const SLOW_LIMIT_START: usize = 10;
pub const SLOW_LIMIT_CUTOFF: usize = 15;
// Ratio of termination time to time since last improvement
pub const SATURATION_FACTOR: u32 = 2;

// Debug field to slow down the pipeline to test termination conditions
pub const SLEEP_TIME: Duration = Duration::from_millis(0);

/// Write to log file if set, otherwise discard
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        {
            use std::io::Write as _;
            let mut guard = $crate::LOG_FILE.lock().unwrap();
            if let Some(ref mut file) = *guard {
                writeln!(file, $($arg)*).ok();
            }
        }
    };
}

pub mod cost;
pub mod ir;
pub mod isel;
pub mod malloc;
