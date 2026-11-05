# Failure policy: dropped candidates and named input stops

Owner decision 40: a generated backend never stops because one candidate fails, and stops only on an invalid input kernel or one outside the input specification.
Today the allocator already returns a reason per candidate (`act_malloc.cc:16-20,182`) and `pipeline.rs:265-268` records it as a drop.
Other candidate-level failures panic, assert, or call `process::exit(1)`; they are listed as D17 to D22 in `bugs.md`.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B41 | Every candidate-level failure is a dropped candidate with a recorded reason (D5, D7, D8, D17 to D21) | B25, B26 | M | Q1 |
| B42 | Input validation and a documented input specification | B5, B46 | M | Q1 |
| B43 | Input formats other than XLA HLO text: detection now, readers staged | B42 | L | Q2 |

## What stops the backend, and what does not

| Event | Outcome | Reason recorded |
| --- | --- | --- |
| An allocator, solver, cost-model, extraction, or rewrite failure on one candidate | Candidate dropped; search continues | In `--log` and in the final no-assembly line |
| Every candidate dropped | Stop with the first drop reason (`pipeline.rs:207-210`) | Yes |
| Invalid input kernel (a syntax or type error in it) | Stop, naming the line and the rule | Yes |
| Input kernel outside the input specification (an operation, element type, or format the backend does not take) | Stop, naming the operation, element type, or format | Yes |
| Input kernel that needs an instruction the backend excluded at generation (B46) | Stop before search, naming the instruction | Yes, from the stored exclusion report |
| Output or log path cannot be written | Stop, naming the path and the system error | Yes |
| A defect in the backend itself | Not a stop condition by design: reported as a bug, tested away | -- |

- A drop reason is one line in the form `<stage>: <what failed>`, with the stages `rewrite`, `extraction`, `allocation`, `cost`.
- `search` returns the drops as data (B29), so a test reads them; `--verbose` prints each.
- The final line counts drops per stage, so a run that dropped most candidates is visible.

## B41 Candidate-level failures as drops

- What: no `panic!`, `unwrap`, `expect`, `assert`, `debug_assert`, `unreachable!`, uncaught C++ exception, or `process::exit` remains on a path that handles one candidate or one rewrite application.
- What: the cost model's failure drops the candidate (D17); `update_best` returns the drop instead of exiting.
- What: a rewrite application that fails (D19) drops that application with a reason, and the e-graph is left as it was before it.
- What: `act_malloc` wraps its body in a catch that turns any exception into a reason (D20).
- What: a loop in extraction that makes no progress (D18) drops the candidate instead of returning the partial graph.
- Preserves: every existing reason text (`act_malloc.cc`, `pipeline.rs:207-210`); the no-assembly line and exit status 1 when every candidate dropped.
- Benefits: one failing candidate or rewrite never ends a compile that has other candidates.
- Caveats: a catch-all hides defects; every caught failure is logged with its reason, and the test mode (B29) fails a test on a drop that was not expected.
- Fail points: a defect surfacing only as drops; noticed by the drop count per stage in the final line and by the unexpected-drop assertion in tests.
- Order and dependencies: after B25 (C++ error returns) and B26 (`Result` in `PiiGraph` and `merge`); closes the remaining crash paths.
- Verification: `grep -nE 'panic!|unwrap\(|expect\(|assert|process::exit'` over the candidate paths names only input-edge and startup code; a test per D17 to D21 path feeds a failing candidate and the compile still finishes with the others.

## B42 Input validation and the input specification

- What: a short document states the input specification: formats, operations (the table behind the rewrites), element types, shape limits, and layouts accepted.
- What: the reader (B5) validates the kernel against it before any search and stops with a named reason of one of two kinds: `invalid kernel` (malformed or ill-typed) or `outside the input specification` (well-formed, not taken).
- What: the reader also reads the exclusion report stored in the crate (B46) and stops with `KB042`, naming the instruction and its reason, when the kernel needs an excluded instruction.
- What: reader panics on kernel text (D22) become these errors; the `.hlo` extension check (`pipeline.rs:312-318`) becomes a check on content, so a mislabeled file is named by its format.
- Preserves: every existing reader message (`line.rs`, `parser.rs`, `lib.rs`).
- Benefits: the only two reasons for the backend to stop are testable, and a user knows which side to fix.
- Caveats: a kernel with an operation no ISA instruction implements is not outside the input specification; it yields no candidate, so it ends as every-candidate-dropped with that reason, unless an excluded instruction implements it (`KB042`).
- Fail points: an invalid kernel reaching selection and dropping every candidate with an unhelpful reason; noticed by a fuzz test of mutated kernels that must end in a named reader error or a compile.
- Order and dependencies: after B5 (annotation-checking reader) and B46; uses the coverage rows for the backend's input.
- Verification: one test per error kind, including a kernel that needs an excluded instruction; a mutation test over the example kernels never panics.

## B43 Input formats not supported today

- Today the backend reads XLA HLO text only (`isel/initializer/lib.rs:36`, `parse_hlo_module_to_egraph`); README names XLA HLO as the one input.
- Decision 40 names four formats: StableHLO, XLA HLO, torch-mlir, torch.fx.

| Format | Today | Plan |
| --- | --- | --- |
| XLA HLO (text) | supported | Stays; the input specification (B42) documents it |
| StableHLO | not supported | Detected and named; reader staged first, since its operations map onto the HLO operation table |
| torch-mlir | not supported | Detected and named; reader staged second, through its StableHLO lowering |
| torch.fx | not supported | Detected and named; reader staged third, graph nodes mapped to the operation table |

- What (Q1, size S): detect the format from content; stop with `outside the input specification: <format> is not supported`.
- What (Q2): one reader per format producing the same kernel record as the HLO reader (B5), so selection is unchanged; operations a format has and the table lacks are named in the input specification.
- Preserves: the HLO reader and its goldens.
- Benefits: a user who passes a StableHLO file gets the format named, not a parse error from the HLO reader.
- Caveats: a reader needs a dependency or a hand-written parser; a new dependency is asked before it lands (engineering rules); format versions drift.
- Fail points: a format reader accepting a dialect it silently misreads; noticed by a differential test against the HLO form of the same kernel on the oracle.
- Order and dependencies: after B42; the Q2 readers also depend on the operation table of B6 and B8.
- Verification: per format, a kernel in that format and in HLO compile to equal assembly; an unsupported format names itself.
