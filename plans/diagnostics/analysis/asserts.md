# Asserts and panics used where an error belongs

## Can fire on user input

| Where | Mechanism | What reaches it | Fix |
| --- | --- | --- | --- |
| `act-backend/act_backend/generic/cpp/malloc/src/parser.cc:316` | `cerr` plus `assert(false && "unreachable")` | A tensor operator the allocator has no case for; the word comes from the kernel through the rewrite metadata | Return a reason (B25) |
| `src/parser.cc:13-14` | `cerr` plus `assert(false)` | A token that is not an integer in a candidate line | Return a reason |
| `src/constraint.cc:112-187` | `cerr` plus `assert(false)` in seven branches | A tensor graph built from the kernel with no uses or no defining instruction | Return a reason |
| C++ build configuration (`CMakeLists.txt`, `generic/build.rs:59`) | `CMakeLists.txt` sets no `CMAKE_BUILD_TYPE`, so asserts are live; a Release build removes every one of them and the helper returns `0`, `{0,0}` or `nullptr` (`parser.cc:15`, `constraint.cc:116`, `parser.cc:189`) | A build type changes behavior | Never rely on `assert` for a check |
| `generic/src/isel/extractor/fast.rs:191` | `debug_assert!(false, ...)` | An extraction loop with no progress; compiled out by `cargo build --release` (`generator.py:182`), so the loop breaks with no message | Return `Err` |
| `act-oracle/taidl_to/generic/utils.py:80` | `ZeroDivisionError(env)` as control flow | A divisor of 0 at some iteration | Raise the coded error where it is detected |
| `act-oracle/taidl_to/generic/utils.py:760` | `raise Unroll` as control flow | A held value that changes over a loop | Keep; it is not a diagnostic, but name it so a grep does not mistake it |

## Internal, but reachable from a malformed spec if a front-end check is missing

| Where | Mechanism | Reached when |
| --- | --- | --- |
| `isel/rewrites/ir2ir_rewrites.rs` (40 `assert_eq!`, `:73`, `:197`) | `assert_eq!`, `unreachable!`, `panic!` | A rewrite matches a pattern with a different class count; a kernel the reader accepted |
| `ir/egraph.rs:140-149` (ten `expect`) | `expect("... needs metadata!")` | A rewrite emits a word without metadata |
| `ir/egraph.rs:407` | `panic!("internal error: a rewrite joins the values ...")` | Two shapes in one e-class; reachable from a kernel whose shapes disagree, which the reader checks only for `slice` and `concatenate` (`nodes.rs:319-431`) |
| `isel/rewrites/applier.rs:27, :41, :55` | `panic!("No precondition function for rewrite {name}!")` | The generated match arms miss a rewrite |
| `ir/pii.rs:36-48` | `panic!` | A candidate node without buffers |
| `isel/extractor/utils.rs:54` | `panic!` | A constant e-class without an operator-only term |
| `isel/initializer/nodes.rs:30` | `unreachable!("binary operation {op}")` | An operator the reader accepted |

## Panics that stand for environment errors

- `generic/build.rs:55-74`, `ir/metadata.rs:24-26`, `ir/pii.rs:70-71`, `pipeline.rs:56, :100, :198, :308` use `expect` or `panic!` for file and process failures.
- These print a Rust panic message and exit status 101, unlike the explicit exits at status 1.

## Reading

- Rust distinguishes user errors well: kernel faults are `Err(String)` all the way to `pipeline.rs:449` and become `Error: ...` with exit 1.
- The C++ allocator mixes returned reasons (good) with `assert(false)` for the same class of fault.
- Python never uses `assert` for a check in the generators.
