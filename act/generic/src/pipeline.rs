#[macro_use]
extern crate backend;

use std::cmp::min;
use std::env;
use std::path::PathBuf;
use std::process;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;

use egg::{Runner, StopReason};

pub use backend::{ir, isel, malloc};
pub use backend::{
    N, PROCESSED, SATURATION_FACTOR, SLEEP_TIME, SLOW_LIMIT_CUTOFF, SLOW_LIMIT_START, TIME_LIMIT,
    LOG_FILE,
};

fn print_help(program_name: String) {
    if program_name == "cargo" {
        // When run via `cargo run -- ...`, the first argument is "cargo"
        println!("Usage: cargo run -- --input <hlo_path> --output <asm_path> [--log <log_dir>]");
    } else {
        println!(
            "Usage: {} --input <hlo_path> --output <asm_path> [--log <log_dir>]",
            program_name
        );
    }
    println!();
    println!("Description:");
    println!("  This program compiles an .hlo file into an assembly code.");
    println!("  Candidate assembly codes are logged in the specified log directory.");
    println!("  The final assembly code is chosen based on performance cost.");
    println!();
    println!("Options:");
    println!("  --help       Print this help message");
    println!("  --input      Specify the input .hlo file path");
    println!("               (required, must have .hlo extension)");
    println!("  --output     Specify the output assembly file path");
    println!("               (required, will be created/overwritten)");
    println!("  --log        Specify the log directory path");
    println!("               (optional, defaults to /tmp/log if not provided)");
    println!("  --stats      Print performance statistics");
    println!("  --stop-at-first  Stop after finding the first valid assembly");
    println!("               (fast extractor only, useful for profiling)");
    println!();
}

#[derive(Clone)]
struct AsmCandidate {
    path: PathBuf,
    cost: i32,
    timestamp: Duration,
}

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

static SHOW_STATS: AtomicBool = AtomicBool::new(false);
static STOP_AT_FIRST: AtomicBool = AtomicBool::new(false);
static STAT_CANDIDATES: AtomicUsize = AtomicUsize::new(0);
static STAT_ITERATIONS: AtomicUsize = AtomicUsize::new(0);
static STAT_EGRAPH_NODES: AtomicUsize = AtomicUsize::new(0);
static BEST: LazyLock<Mutex<Option<AsmCandidate>>> = LazyLock::new(|| Mutex::new(None));
static LOG_PATH: LazyLock<Mutex<Option<PathBuf>>> = LazyLock::new(|| Mutex::new(None));

fn print_stats(total_time: Duration) {
    if !SHOW_STATS.load(Ordering::Relaxed) { return; }

    // Flush log file before reading it
    {
        let mut guard = LOG_FILE.lock().unwrap();
        if let Some(ref mut f) = *guard {
            use std::io::Write;
            f.flush().ok();
        }
    }

    println!("[STATS] total_time={:.2?}", total_time);
    println!("[STATS] rewrite_iterations={}", STAT_ITERATIONS.load(Ordering::Relaxed));
    println!("[STATS] egraph_nodes={}", STAT_EGRAPH_NODES.load(Ordering::Relaxed));
    println!("[STATS] candidates_generated={}", STAT_CANDIDATES.load(Ordering::Relaxed));
    match BEST.lock().unwrap().as_ref() {
        Some(b) => {
            println!("[STATS] best_cost={}", b.cost);
            println!("[STATS] best_found_at={:.2?}", b.timestamp);
        }
        None => {
            println!("[STATS] best_cost=NONE");
        }
    }

    // Parse log file for per-module timing
    if let Some(ref log_path) = *LOG_PATH.lock().unwrap() {
        if let Ok(content) = std::fs::read_to_string(log_path) {
            use std::collections::HashMap;
            let mut module_totals: HashMap<String, (u64, usize)> = HashMap::new();

            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("[STATS] module=") {
                    let parts: Vec<&str> = rest.splitn(3, ' ').collect();
                    if let Some(name) = parts.first() {
                        if let Some(time_part) = rest.split("time=").nth(1) {
                            let us = parse_duration_us(time_part.trim());
                            let entry = module_totals.entry(name.to_string()).or_insert((0, 0));
                            entry.0 += us;
                            entry.1 += 1;
                        }
                    }
                }
            }

            let module_order = ["M1_Initializer", "M2_Rewriter", "M3_Extractor",
                                "M4_TopoSort", "M5_CSPGen", "MX_Solver", "M6_Emitter",
                                "MC_CostModel"];
            for name in module_order {
                if let Some((total_us, count)) = module_totals.get(name) {
                    println!("[STATS] {}={:.3}ms ({} calls)", name, *total_us as f64 / 1000.0, count);
                }
            }
        }
    }
}

fn parse_duration_us(s: &str) -> u64 {
    s.trim().strip_suffix("us").and_then(|v| v.parse::<u64>().ok()).unwrap_or(0)
}

fn check_termination(start: &Instant, output_path: &PathBuf) {
    let current_time = start.elapsed();
    let best_snapshot = BEST.lock().unwrap().clone();
    match best_snapshot {
        Some(AsmCandidate {
            path,
            cost,
            timestamp,
        }) => {
            if current_time > timestamp * SATURATION_FACTOR {
                log!(
                    "No improvement for last {:?}, stopping",
                    current_time - timestamp
                );
                log!();
                log!("Total time: {:?}", current_time);
                log!(
                    "Best ASM {:?} with cost {} found at {:?}",
                    path, cost, timestamp
                );

                std::fs::copy(&path, output_path).expect("Failed to copy best ASM to output path");
                log!("Best ASM copied to output path: {}", output_path.display());

                print_stats(current_time);
                process::exit(0);
            } else {
                log!(
                    "Current elapsed: {:?} | Will stop at {:?} if no progress",
                    current_time,
                    min(timestamp * SATURATION_FACTOR, TIME_LIMIT)
                );
            }
        }
        None => {
            log!(
                "Current elapsed: {:?} | Will stop at {:?} if no progress",
                current_time, TIME_LIMIT
            );
        }
    }
}

fn timeout(start: &Instant, output_path: &PathBuf) {
    let current_time = start.elapsed();
    if current_time > TIME_LIMIT {
        log!("Reached overall time limit of {:?}, stopping", TIME_LIMIT);
        log!();
        log!("Total time: {:?}", current_time);
        print_stats(current_time);
        let best_snapshot = BEST.lock().unwrap().clone();
        match best_snapshot {
            Some(AsmCandidate {
                path,
                cost,
                timestamp,
            }) => {
                log!(
                    "Best ASM {:?} with cost {} found at {:?}",
                    path, cost, timestamp
                );

                std::fs::copy(&path, output_path).expect("Failed to copy best ASM to output path");
                log!("Best ASM copied to output path: {}", output_path.display());
            }
            None => {
                log!("Could not find an ASM representation.");

                if output_path.exists() {
                    std::fs::remove_file(output_path)
                        .expect("Failed to remove existing output file");
                }
                log!("No output file created.");
            }
        }
        process::exit(0);
    } else {
        eprintln!(
            "Error: timeout() called at {:?} but overall time limit of {:?} not reached",
            current_time, TIME_LIMIT
        );
        process::exit(1);
    }
}

fn update_best(asm_path: &PathBuf, start: &Instant) -> bool {
    if !asm_path.exists() {
        return false;
    }

    log!("Current elapsed: {:?} | Will stop at {:?} if no progress", start.elapsed(), TIME_LIMIT);
    log!("Starting Phase X: Performance Cost Evaluation for {:?}", asm_path);
    log!();

    let mc_start = Instant::now();
    let cost: i32 = backend::cost::python_bridge(&asm_path);
    log!("[STATS] module=MC_CostModel time={}us", mc_start.elapsed().as_micros());
    let time = start.elapsed();

    log!("New ASM has performance cost {}. ", cost);

    let mut best = BEST.lock().unwrap();
    match best.as_ref() {
        Some(b) => {
            if cost < b.cost {
                log!("Better than the previous best ({}), updating.", b.cost);
                *best = Some(AsmCandidate {
                    path: asm_path.clone(),
                    cost,
                    timestamp: time,
                });
                return true;
            } else {
                log!("Not better than the previous best ({}).", b.cost);
                return false;
            }
        }
        None => {
            log!("First ASM found, setting as best.");
            *best = Some(AsmCandidate {
                path: asm_path.clone(),
                cost,
                timestamp: time,
            });
            return true;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.contains(&"--help".to_string())
        || !args.contains(&"--input".to_string())
        || !args.contains(&"--output".to_string())
    {
        print_help(args[0].clone());
        process::exit(0);
    }

    // Process input file
    let input_index = args.iter().position(|x| x == "--input").unwrap() + 1;
    if input_index >= args.len() {
        eprintln!("Error: Missing file name after --input");
        process::exit(1);
    }

    let hlo_path_arg = &args[input_index];
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let hlo_path = current_dir.join(hlo_path_arg);

    if hlo_path.extension().and_then(|s| s.to_str()) != Some("hlo") {
        eprintln!(
            "Error: Input file '{}' is not an .hlo file.",
            hlo_path.display()
        );
        process::exit(1);
    }

    if !hlo_path.exists() {
        eprintln!("Error: Input file '{}' does not exist.", hlo_path.display());
        process::exit(1);
    }

    log!("Input file: {}", hlo_path.display());

    // Process output file
    let output_index = args.iter().position(|x| x == "--output").unwrap() + 1;
    if output_index >= args.len() {
        eprintln!("Error: Missing file name after --output");
        process::exit(1);
    }
    let output_path_arg = &args[output_index];
    let output_path = current_dir.join(output_path_arg);

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create output directory");
        }
    }

    if output_path.extension().and_then(|s| s.to_str()) != Some("py") {
        eprintln!(
            "Error: Output file '{}' does not have a .py extension.",
            output_path.display()
        );
        process::exit(1);
    }

    if output_path.exists() {
        log!(
            "Warning: Output file '{}' already exists and will be overwritten.",
            output_path.display()
        );
    }

    log!("Output file: {}", output_path.display());

    // Process log directory
    let log_dir_arg: String = if args.contains(&"--log".to_string()) {
        let log_index = args.iter().position(|x| x == "--log").unwrap() + 1;
        if log_index >= args.len() {
            eprintln!("Error: Missing directory after --log");
            process::exit(1);
        }
        args[log_index].clone()
    } else {
        std::env::temp_dir().join(format!("act_log_{}", process::id())).to_string_lossy().to_string()
    };

    let log_dir = std::path::PathBuf::from(&log_dir_arg);
    if log_dir.exists() {
        std::fs::remove_dir_all(&log_dir).expect("Failed to remove existing log directory");
    }
    std::fs::create_dir_all(&log_dir).expect("Failed to create log directory");

    // Set up log file inside the log directory
    let log_file_path = log_dir.join("log.txt");
    {
        let file = std::fs::File::create(&log_file_path)
            .unwrap_or_else(|e| { eprintln!("Error creating log file: {}", e); process::exit(1); });
        *LOG_FILE.lock().unwrap() = Some(file);
        *LOG_PATH.lock().unwrap() = Some(log_file_path.clone());
    }

    // Process --stats flag
    if args.contains(&"--stats".to_string()) {
        SHOW_STATS.store(true, Ordering::Relaxed);
    }

    // Process --stop-at-first flag
    if args.contains(&"--stop-at-first".to_string()) {
        STOP_AT_FIRST.store(true, Ordering::Relaxed);
    }

    log!("Log directory: {}", log_dir.display());
    log!();

    // Start processing the input file
    let start = Instant::now();

    check_termination(&start, &output_path);
    log!("Starting Phase 1: Instruction Selection...");
    log!();

    check_termination(&start, &output_path);
    log!("Starting Phase 1 Module 1: E-Graph Initializer...");
    log!();

    let m1_start = Instant::now();
    let (init_egraph, hbm_offsets, root, inputs, metadata) =
        isel::initializer::parse_hlo_module_to_egraph(&hlo_path).unwrap();
    log!("[STATS] module=M1_Initializer time={}us", m1_start.elapsed().as_micros());

    log!("HBM Offsets: {:?}", hbm_offsets);
    log!("Root ID: {:?}", root);
    log!("Inputs: {:?}", inputs);
    log!();

    let metadata_path = log_dir.join("metadata.json");
    metadata.save(&metadata_path);

    let mut limit: usize = SLOW_LIMIT_START;

    let rules = isel::rewrites::get_rewrites();
    let inputs_for_hook = inputs.clone();
    let hbm_offsets_for_hook = hbm_offsets.clone();

    let output_path_for_hook = output_path.clone();
    let log_dir_for_hook = log_dir.clone();
    let metadata_path_for_hook = metadata_path.clone();

    let runner = {
        // clone the Rcs
        Runner::default()
            .with_egraph(init_egraph)
            .with_node_limit(50000)
            .with_iter_limit(1000)
            .with_time_limit(TIME_LIMIT)
            .with_hook(move |runner| {
                PROCESSED.lock().unwrap().clear();
                STAT_ITERATIONS.store(runner.iterations.len(), Ordering::Relaxed);
                STAT_EGRAPH_NODES.store(runner.egraph.total_number_of_nodes(), Ordering::Relaxed);
                if runner.iterations.len() % N == 0 && runner.iterations.len() > 0 {
                    check_termination(&start, &output_path_for_hook);
                    log!(
                        "Starting Phase 1 Module 3: Graph Extractor (limit {})",
                        limit
                    );
                    log!();

                    let m3_start = Instant::now();
                    let stop_at_first = STOP_AT_FIRST.load(Ordering::Relaxed);
                    let piis = isel::extractor::extract(
                        &mut runner.egraph.clone(),
                        root,
                        &inputs_for_hook,
                        &hbm_offsets_for_hook,
                        limit,
                        stop_at_first,
                    );
                    log!("[STATS] module=M3_Extractor iteration={} time={}us", runner.iterations.len(), m3_start.elapsed().as_micros());
                    limit += 1; // Increment limit to allow for more extraction next time

                    for pii in piis {
                        check_termination(&start, &output_path_for_hook);
                        log!("Starting Phase 2 for PII #{}", STAT_CANDIDATES.load(Ordering::Relaxed));
                        log!();

                        // Phase 2
                        let pii_path =
                            log_dir_for_hook.join(format!("{}.pii", STAT_CANDIDATES.load(Ordering::Relaxed)));
                        pii.save(&pii_path);

                        let asm_path =
                            log_dir_for_hook.join(format!("{}.py", STAT_CANDIDATES.load(Ordering::Relaxed)));

                        match backend::malloc::cpp_bridge(&pii_path, &metadata_path_for_hook, &asm_path) {
                            1 => {
                                if stop_at_first {
                                    std::fs::copy(&asm_path, &output_path_for_hook).expect("Failed to copy ASM");
                                    log!("First valid ASM found, stopping (--stop-at-first)");
                                    print_stats(start.elapsed());
                                    process::exit(0);
                                }
                                update_best(&asm_path, &start);  // Check cost and update best if necessary
                            }
                            0 => { log!("Phase 2: no solution for PII #{}", STAT_CANDIDATES.load(Ordering::Relaxed)); }
                            _ => {
                                eprintln!("Error: Phase 2 failed for PII #{}", STAT_CANDIDATES.load(Ordering::Relaxed));
                                process::exit(1);
                            }
                        }
                        log!();

                        STAT_CANDIDATES.fetch_add(1, Ordering::Relaxed);
                    }

                    check_termination(&start, &output_path_for_hook);
                    log!("Completed Phase 2: Memory Allocation, returning to Phase 1: Instruction Selection");
                    log!();
                }

                // Log M2 timing: per-iteration = total_time[i] - total_time[i-1]
                if !runner.iterations.is_empty() {
                    let n = runner.iterations.len();
                    let curr = runner.iterations[n - 1].total_time;
                    let prev = if n >= 2 { runner.iterations[n - 2].total_time } else { 0.0 };
                    let delta_us = ((curr - prev) * 1_000_000.0) as u64;
                    log!("[STATS] module=M2_Rewriter iteration={} time={}us", n, delta_us);
                }

                check_termination(&start, &output_path_for_hook);
                log!(
                    "Starting Phase 1 Module 2: Rewrite Applier (iteration {})",
                    runner.iterations.len() + 1
                );
                log!();
                sleep(SLEEP_TIME);

                Ok(())
            })
            .run(&rules.clone())
    };

    // Logic based on the stop reason:
    // 1. TimeLimit: No more extraction, just return.
    // 2. NodeLimit, Saturated: Run extraction until time limit is hit.
    // 3. IterationLimit: Should not have happened. Recheck if there is a default limit.
    // 4. Other: Should not have happened. Requires investigation.

    match runner.stop_reason.as_ref().unwrap() {
        StopReason::TimeLimit(_) => {
            log!("Info: Reached time limit. No further extraction.");
            log!();
        }
        StopReason::NodeLimit(_) | StopReason::Saturated => {
            log!("Info: Reached node limit or saturated. Running extraction until time limit is hit.");
            log!();

            while start.elapsed() < TIME_LIMIT {
                check_termination(&start, &output_path);
                log!(
                    "Starting Phase 1 Module 3: Graph Extractor (limit {})",
                    limit
                );
                log!();

                let m3_start = Instant::now();
                let stop_at_first = STOP_AT_FIRST.load(Ordering::Relaxed);
                let piis = isel::extractor::extract(
                    &mut runner.egraph.clone(),
                    root,
                    &inputs,
                    &hbm_offsets,
                    limit,
                    stop_at_first,
                );
                log!("[STATS] module=M3_Extractor iteration=post time={}us", m3_start.elapsed().as_micros());
                limit += 1; // Increment limit to allow for more extraction next time

                for pii in piis {
                    check_termination(&start, &output_path);
                    log!("Starting Phase 2 for PII #{}", STAT_CANDIDATES.load(Ordering::Relaxed));
                    log!();

                    // Phase 2
                    let pii_path = log_dir.join(format!("{}.pii", STAT_CANDIDATES.load(Ordering::Relaxed)));
                    pii.save(&pii_path);

                    let asm_path = log_dir.join(format!("{}.py", STAT_CANDIDATES.load(Ordering::Relaxed)));
                    match backend::malloc::cpp_bridge(&pii_path, &metadata_path, &asm_path) {
                        1 => {
                            if stop_at_first {
                                std::fs::copy(&asm_path, &output_path).expect("Failed to copy ASM");
                                log!("First valid ASM found, stopping (--stop-at-first)");
                                print_stats(start.elapsed());
                                process::exit(0);
                            }
                            update_best(&asm_path, &start);
                        }
                        0 => { log!("Phase 2: no solution for PII #{}", STAT_CANDIDATES.load(Ordering::Relaxed)); }
                        _ => {
                            eprintln!("Error: Phase 2 failed for PII #{}", STAT_CANDIDATES.load(Ordering::Relaxed));
                            process::exit(1);
                        }
                    }
                    log!();

                    STAT_CANDIDATES.fetch_add(1, Ordering::Relaxed);
                }

                check_termination(&start, &output_path);
                log!("Completed Phase 2: Memory Allocation, returning to Phase 1: Instruction Selection");
                log!();

                sleep(SLEEP_TIME);
            }
            log!("Info: Reached time limit. No further extraction.");
        }
        StopReason::IterationLimit(_) => {
            log!("Info: Reached iteration limit. Continuing with extraction.");
            log!();
        }
        StopReason::Other(_) => {
            eprintln!("Error: Stopped for an unexpected reason. Requires investigation.");
            process::exit(1);
        }
    }

    timeout(&start, &output_path);
}
