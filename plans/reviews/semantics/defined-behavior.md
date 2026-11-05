# Review: defined behavior and numerics

Independent review of whether the plans deliver owner decisions 16 (no undefined behavior) and 17 (an in-place mark reuses the input's storage) across the spec, the formal semantics, act-oracle, and act-backend.
Inputs: `minutes/rulings.md`, `semantics/`, `language/spec-draft/`, `oracle/plan/fidelity.md`, `reviews/oracle/critic.md`, `backend/plan/`, and the three code repos read-only.

## Verdict

The plans do not yet deliver decision 16 consistently.
The tech-lead rulings and the semantics track ruled the same questions differently in three places: the state view of constraints (R7), integer overflow in expressions (R56), and the meaning of an input-output overlap (R5).
Each of the three disagreements is a blocker, because each generator would have to choose a side.
Every S1.3 row has a candidate outcome somewhere, except a float or integer `convert` out of range, which no ruling covers.
The numerics norms in `dynamics.md` are close to right but claim behavior the oracle does not have today: per-op rounding at narrow floats (XLA excess precision), and possibly kept denormals.
R12 (zero-filled buffers) is cheap for the oracle but is a promise real hardware does not keep, and nothing in the backend plan addresses it.
Fault detection is cheap: every region address is a function of host integers (attributes and state), so the oracle can check every fault on the host, at zero device cost.
This holds only if every data-dependent case stays a defined value and never a fault, which the plans should state as the norm.
Read-before-write and aliasing compose correctly once R5 is the only aliasing rule and other overlaps are defined, not rejected.

## Findings

| ID | Severity | Where | Finding | Proposed norm or change |
| --- | --- | --- | --- | --- |
| DB1 | blocker | R7; instructions.md:36,43; dynamics.md:122; holes.md H3, H11 | R7 makes constraints read the post-update state, but S5.3, S5.4 step 1, `dynamics.md` step 1 (`rho_0`), and H11 (oracle checks constraints on the pre-state) all read the pre-state. | One state view: every constraint, start, size, and backtick expression reads the state after all updates; rewrite S5.3, S5.4, `dynamics.md` steps 1-3, and H3, H11 to match. |
| DB2 | blocker | R56; holes.md H4; dynamics.md:106,113; expressions.md:89; fidelity.md:45 (O11); R38 | R56 makes intermediates exact and faults only a final value outside i64, while H4, `dynamics.md`, S6.2, and O11 fault on any intermediate. The backend evaluates in i64, so R56 forces it to emulate exact arithmetic. | Re-rule R56: any intermediate outside i64 is a fault. Both generators meet it with checked i64 arithmetic (Python ints in the oracle, `checked_*` in Rust and C++). Align S6.2, H4, O11, and R38 to the same sentence. |
| DB3 | blocker | R5; holes.md H1; semantics/rulings.md:64-69; instructions.md:50-53; fidelity.md:27-37 (O10) | R5 removes `set_inplace` and derives aliasing from identical region text, but H1 and the semantics ruling still use the flag and reject an exact overlap without it and any partial overlap. O10 still asks L what an in-place input holds and offers NaN poisoning. | Aliasing is R5 alone. Any other input-output overlap is defined by read-before-write and is not a fault. An ISA whose hardware cannot overlap writes a constraint (R58). Rewrite H1 and the semantics ruling; reduce O10 to "drop `set_inplace`". |
| DB4 | major | R12; templates.txt:470 (oracle zero fill); backend/plan (no item) | R12 promises zero-filled buffers. The oracle meets it (`sh_zeros` in PROLOGUE_INIT), but hardware scratchpads start with stale data, so an oracle pass hides a kernel that reads unwritten storage. R12 also says "every buffer" while `d0` is the caller's `hbm` argument (H8). | A read of an on-chip element no earlier instruction of the kernel wrote is a fault. `d0` holds the kernel's input. The oracle tracks written intervals per buffer on the host, with no device cost; the backend may assume it, since its dataflow writes before it reads. |
| DB5 | major | R4, R10 ("may assume"); holes.md:54 ("never emits"); R21; backend/plan | The backend chooses `@a.` values and placements, so for bounds and constraints it is a producer that must guarantee them, not assume them. R21 (aliasing as an allocator equality constraint) has no backend plan item; B11 (extractor.md:22-25) only reshapes the copy rule. | State it once: the backend never emits an issue that faults, and lists each precondition it cannot see (state before R27, `|d0|`) in its coverage doc. Add an S item for R21 next to B11. |
| DB6 | major | R8; holes.md H6; dynamics.md:70; fidelity.md:74 (O13) | R8 bounds the error relative to the exact result rounded to the result type. Under cancellation that bound is unmeetable by hardware that accumulates at bf16. "Accumulate in the result type" is a procedure no test can observe. R8 says "reduce" though max and min are exact. O13 promises exact comparison for every non-transcendental word. | Norm for float `dot`, `reduce_add`, `reduce_multiply`: `|r - exact| <= n * u * sum|terms|` plus one rounding, with `n` the term count and `u` the unit roundoff of the result type. Drop the accumulation sentence. Restrict R8 to those three words. O13 compares them by this bound. |
| DB7 | major | dynamics.md:69 (float exact); critic.md:43; oracle (no `XLA_FLAGS`) | The float exact class needs each op rounded to its result type, but XLA's default `xla_allow_excess_precision=true` lets bf16 and f16 chains stay at f32. No plan item sets the flag. | Every float operation rounds its result to its result type once. The oracle sets `--xla_allow_excess_precision=false` before jax initializes, and a test checks a bf16 chain rounds per op. |
| DB8 | major | dynamics.md:69; holes.md H6 ("denormals kept") | No test backs "denormals kept". The XLA CPU client may run under a flush-to-zero scope, and many accelerators flush too. | Fix one norm after a test on the pinned jaxlib: keep denormals if XLA keeps them, else flush every float operation's denormal input and result to a zero of the same sign. Never "either". |
| DB9 | major | overview.md:39 (S1.3 convert row); dynamics.md:69; no ruling | No ruling covers `convert` out of range. `dynamics.md` puts `convert` in the float exact class with nearest-even rounding, which is wrong for float to int: StableHLO truncates, and calls out-of-range "TBD". | Float to float and int to float: nearest-even, overflow to signed infinity, or NaN where the type has none (`f8e4m3fn`). Float to int: truncate toward zero, saturate, NaN to 0. Int to int: wrap. To `pred`: nonzero is true. Test each against the pinned XLA; a saturating ISA writes `clamp` first. |
| DB10 | major | operations.md S8.1 (`dynamic_slice`, `dynamic_update_slice`, `gather`); critic.md:46 | These words take start indices from tensor data, the only data-dependent addresses in the language. StableHLO clamps them, but neither S1.3 nor `dynamics.md` says so. The critic proposes a device-side fault flag for data-dependent errors such as a zero divisor. | Norm: tensor data never faults. In-semantics indices clamp as StableHLO does, and divisors take the integer divide row's values. Faults then depend only on attributes, state, and region text, so the oracle checks all of them on the host. |
| DB11 | minor | dynamics.md:114; expressions.md:97; holes.md H4, H5 | A REQUIRE failure inside a constraint "does not hold", while R55 calls every such case a fault: two words for one outcome. | Any REQUIRE failure is a fault (R55); the backend's reading is "never selected". |
| DB12 | minor | dynamics.md:68-69 | The integer rows leave several cases open: `min % -1`, integer `power` with a negative exponent, `abs` and `negate` of the minimum, `clamp` with `lo > hi`, and a negative shift amount. Integer `dot` is in no class. | Add: `min % -1 = 0`; integer `power` takes XLA's value; `abs` and `negate` wrap; `clamp` is `min(max(x, lo), hi)`; the shift amount is read unsigned; integer `dot` is exact with wraparound at the result type. |
| DB13 | minor | semantics/rulings.md:75; utils.py:441 | The text says a float sum over a size-1 dimension gives the element itself. That is false for -0.0, since `-0.0 + +0.0 = +0.0`. | Keep +0.0 as the sum's start (matches XLA) and state that an all-negative-zero sum is +0.0. |
| DB14 | minor | dynamics.md:69 ("quiet NaN"); S8.4 | NaN sign and payload are left open, and `bitcast_convert` exposes them. The sign of zero from `maximum` and `minimum` on (-0, +0) is not stated. | Any NaN equals any NaN, and the payload is not part of the meaning; say so as the one non-bitwise point of the exact class. `maximum` and `minimum` follow IEEE 754-2019: NaN propagates and -0 < +0. Verify by test. |
| DB15 | minor | R57; holes.md H15 | "Must fit the element type" is undefined for a float literal that rounds to infinity or underflows. | A finite decimal that rounds to infinity (NaN at `f8e4m3fn`) is rejected; underflow rounds normally, subject to DB8. |
| DB16 | minor | dynamics.md:70 (approximate class) | A per-word tolerance does not pin special values such as `exp(-inf)`, `log(0)`, or `tanh(inf)`. | Special values of every approximate word are exact: inf, zero, and NaN results match the IEEE function exactly. |
| DB17 | minor | R7 reason | R7 cites `instruction.py:216-218`, which is placement-mark code (`/home/user/taidl/taidl/instruction.py:205-222`), not the state read order. | Cite `instruction_codegen.py:732-737` in act-oracle. |
| DB18 | minor | R4; holes.md H8; overview.md:32-48 | R4 names out-of-bounds regions but not a size below 1, and S1.3 still lists 15 undefined or implementation-defined rows with no item to empty it. | R4 covers size below 1. Replace S1.3 with a fault list plus a pointer to the numerics norm table. |
| DB19 | minor | R28; holes.md H3; fidelity.md O11 | H3 asks the oracle to reject rather than unroll looped starts, while R28 unrolls every loop. O11 exists only if the R28 spike fails. | Delete the H3 remedy. Make O11 conditional on the R28 spike, enforcing DB2's rule. |

## Fault detection cost in the oracle

- Region bounds, output overlap, expression overflow, divisor 0, and constraints: host integers per call, already enumerated per iteration (`utils.py:698-712`), so the cost is trace time only.
- Under R28 every loop is unrolled, so each check runs once per issued instruction, linear in issues times regions.
- Unwritten-read tracking (DB4): one interval set per on-chip buffer on the host, with no device cost.
- No fault needs a device-side flag once DB10 holds; a device-side check would cost one `reduce_or` and a host sync per call.

## Run-time cases and their single norm

| Case | Proposed single norm |
| --- | --- |
| Region start below 0, size below 1, or end past its dimension or `|d0|` | Fault |
| Two output regions of one issue overlap | Fault |
| Input and output region texts identical | Alias: the output reuses the input's storage; the value is by read-before-write |
| Input and output overlap otherwise | Defined by read-before-write; not a fault |
| A constraint does not hold | Fault |
| An expression intermediate leaves i64, including a state update | Fault |
| A computed `/` or `%` divisor is 0 | Fault |
| Region value element count differs from the region's at issue | Fault |
| A backtick constant outside its element type | Fault |
| A read of an on-chip element no earlier instruction wrote | Fault |
| Storage no output region writes | Unchanged |
| State seen by constraints, starts, sizes, and backticks | The state after all updates, which run in order |
| Initial value of `reduce_<r>` | The identity of r at the type; a float sum starts at +0.0 |
| Integer elementwise ops and reductions | Two's complement, wrapping at the type's width |
| Integer `divide` and `remainder` by 0, `min / -1`, `min % -1` | -1 (signed) or all ones (unsigned); `x % 0 = x`; `min`; 0 |
| Shift by at least the width | 0, or the sign fill for an arithmetic right shift; the amount is read unsigned |
| Integer `dot` | Exact, wrapping at the result type |
| Float exact-class operation | IEEE round to nearest even at the result type, once per op, no excess precision |
| NaN | Propagates; any NaN equals any NaN |
| Signed zero in `maximum` and `minimum` | IEEE 754-2019: -0 < +0 |
| Denormal input or result | One choice fixed by test on the pinned XLA (DB8) |
| Float `dot`, `reduce_add`, `reduce_multiply` | Within `n * u * sum|terms|` plus one rounding |
| Transcendental words | Within the per-word ulp bound; special values exact |
| Float to float, int to float `convert` | Nearest even; overflow to infinity, or NaN where none exists |
| Float to int `convert` | Truncate toward zero, saturate, NaN to 0 |
| Int to int, any to `pred` `convert` | Wrap; nonzero is true |
| In-semantics `dynamic_slice`, `dynamic_update_slice`, `gather` index | Clamped as in StableHLO |
| `clamp` with `lo > hi` | `min(max(x, lo), hi)` |
| `bitcast_convert` byte order, sub-byte packing | Little-endian; sub-byte elements fill a byte from its low bits |
| Float literal | Rounded once to nearest even from the decimal; rounding to infinity is rejected |
| `compare` without `type` | FLOAT, SIGNED, or UNSIGNED by element type; UNSIGNED for `pred` |
