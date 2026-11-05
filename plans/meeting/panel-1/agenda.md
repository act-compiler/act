# Panel 1 agenda: defined behavior and numerics

Scope: faults, accumulation, literals, integer arithmetic, aliasing and in-place semantics, partial overlap, state read order, initial and unwritten contents, convert out of range, NaN and denormals, index clamping.
Moderator: Fable.
Debaters: Sonnet (oracle implementer), Opus (semantics and spec author), Fable (hardware architect).

## Binding constraints for every question

- Owner decision 16: no undefined behavior in v3; every run-time violation is a defined error or a defined behavior under one simple norm that is easy to explain.
- Owner decision 17: an in-place mark means the output reuses the marked input's storage.
- Owner decision 14: the owner's papers first when precise, otherwise StableHLO and MLIR.
- Owner decision 10: breaking changes are allowed, batched into one major release.
- Owner decisions 19 to 21: buffers and state, region, the `@s.` prefix.
- A draft answer (R-number) is one option among others and carries no default status.
- Source tags: DB = reviews/semantics/defined-behavior.md; SC = reviews/semantics/checker.md; H = semantics/holes.md; RA = reviews/rulings/audit.md; O = oracle plan; S = language/spec-draft section; R = minutes/rulings.md.

## Order

- Q1.1 to Q1.3 are the three blockers DB1 to DB3 (also SC1 to SC3).
- Q1.4 to Q1.12 follow, grouped by topic.
- Minor findings that need no debate are in the final list "For the authors".

### Q1.1: Which state do constraints, starts, sizes, and backtick expressions read?

- Context: DB1 (blocker).
  R7 says starts, sizes, and constraints all read the state after the instruction's updates.
  The spec draft (S5.3, S5.6 step 1) reads user constraints before the updates and placement-mark constraints after them; dynamics.md step 1 and H11 read constraints on the pre-state; the oracle's `CHECK_CONSTRAINT` also reads the pre-state, while its starts read the post-state.
- Option A: every constraint, start, size, and backtick expression reads the state after all updates (R7, DB1, SC1); an update fault precedes any constraint test.
- Option B: user constraints read the pre-state; placement-mark constraints read the post-state, as the spec draft has it (RA R7); after R58 only user constraints remain, all on the pre-state.
- Option C: the pre-state everywhere, including starts and sizes (H3 "Q2 pre-state everywhere", F4).
- Option D: split by role: every constraint is a precondition on the pre-state, while region addresses and the semantics read the post-state (dynamics.md steps 1 to 3); this equals option B once R58 removes placement marks.
- Evidence: DB1; SC1; RA R7 (SME `@s.sm` guard, casestudies/sme.md:20-30; update rollback under R55); holes.md H3, H11; spec-draft S5.3, S5.4, S5.6; semantics/dynamics.md steps 1 to 3; oracle `instruction_codegen.py:732-737` and `templates.txt:353`.
- Binding: OD16 (one state view must be easy to explain); R58 removal of placement marks changes which constraints exist.

### Q1.2: Does an expression fault on any intermediate outside i64, or only on its final value?

- Context: DB2 (blocker).
  R56 evaluates exactly and faults only when the final value of a start, size, dimension, operation attribute, comparison operand, or state update lies outside signed 64 bits (spec-draft S6.2, S1.3.1).
  H4, dynamics.md, and O11 fault on any intermediate outside i64.
  The backend generates checked i64 arithmetic per operator (`expression_writer.py:72,94`); the oracle evaluates in Python integers and warns today.
- Option A: exact intermediates; one check on the final value of each expression text (R56, SC3).
- Option B: every intermediate outside i64 is a fault; both generators use checked i64 (Python ints with a range check per step in the oracle, `checked_*` in Rust and C++) (DB2, H4, RA R56).
- Evidence: DB2; SC3; RA R56 (backend would need i128 or bignums under A); holes.md H4; dynamics.md "Expression evaluation"; spec-draft S6.2, S1.3.1; oracle fidelity.md O11, O24; R38; R28 unroll spike (DB19).
- Binding: OD16; the same sentence must hold in S6.2, H4, O11, and R38.
- Cost point to weigh: an A-conformant backend must emulate exact arithmetic or prove a bound; a B-conformant oracle must check each step of an unrolled expression.

### Q1.3: What is aliasing, and what happens on other overlap between an input and an output?

- Context: DB3 (blocker).
  R5 derives aliasing from identical region text and removes `set_inplace`.
  H1 and semantics/rulings.md keep the flag, reject an unmarked exact overlap, and reject any partial overlap.
  The spec draft S5.5 keeps `set_inplace`, requires equal sizes by tokens, and faults at issue when the starts differ (S1.3.6).
  Dynamics.md allows any overlap by read-before-write with no REQUIRE.
  O10 still asks what an in-place input holds after the call.
- Option A: an input and an output alias exactly when their region texts are identical; `set_inplace` is removed; any other overlap is defined by read-before-write and is not a fault; hardware that cannot overlap writes a constraint (R5, DB3, R58).
- Option B: alias by identical text; any other overlap at issue, including equal regions spelled differently and partial overlap, is a fault checked by the oracle and never emitted by the backend (RA R5, RA2).
- Option C: keep `set_inplace` as the mark; the flagged input must equal output region 0 in buffer and size by text and in starts at issue; an unflagged exact overlap or any partial overlap is rejected or faults (H1, semantics/rulings.md, spec-draft S5.5).
- Option D: alias is decided by the evaluated regions at issue (same buffer, equal starts and sizes after evaluation), not by text; the rest as A or B (gap raised in RA R5).
- Common to all options: the value is read-before-write, so a read after the issue of a marked input sees the output (OD17).
- Evidence: DB3; SC2; SC4; RA R5, R21, RA2; holes.md H1, H2; spec-draft S5.2, S5.5, S1.3.4, S1.3.6; oracle fidelity.md O10; `act-backend/tests/test_extraction.py:405-414,507`; `examples/mini_trn/mini_trn.py:97-99`; R21 and B11 (allocator equality plus copies).
- Binding: OD17; OD16; R58 interacts with the fallback of writing a constraint.

### Q1.4: What do on-chip buffers hold before the first write, and is a read of unwritten storage legal?

- Context: DB4.
  R12 says storage no output region writes is unchanged and every buffer starts zero-filled.
  The oracle zero-fills on-chip buffers today (`templates.txt:470`); real scratchpads start with stale data, so an oracle pass can hide a kernel that reads unwritten storage.
  `d0` holds the caller's `hbm` contents (spec-draft S5.6, H8); R12 as worded covers every buffer.
- Option A: every on-chip buffer starts zero; `d0` holds the `hbm` contents; an unwritten element keeps its value (R12 limited to on-chip, spec-draft S1.3.11, S5.6).
- Option B: option A plus a backend duty: emit zeroing, or reject a kernel that reads unwritten storage (RA R12).
- Option C: a read of an on-chip element that no earlier instruction of the kernel wrote is a fault; the oracle tracks written intervals per buffer on the host (DB4).
- Evidence: DB4; RA R12; SC13 (`Mem_0`, `Reg_0`, `|d0|` missing from the machine state); holes.md H8; spec-draft S5.6, S4.3; `templates.txt:470`; no backend plan item exists.
- Binding: OD16 (no undefined read).
- Cost point to weigh: option C costs host-side bookkeeping only; option A promises behavior some hardware does not keep.

### Q1.5: What is the norm for float `dot`, `reduce_add`, and `reduce_multiply` accumulation?

- Context: DB6, SC24.
  R8 says accumulate in the result element type and land within a per-word tolerance of the exact result rounded to that type.
  S1.3.15 and S8.4 leave accumulation open.
  The oracle review response notes that CPU goldens are bit-exact today and that XLA CPU does not accumulate bf16 in bf16.
- Option A: accumulate in the result type; result within a per-word tolerance of the exact result rounded to the type; wider accumulation is written with explicit `convert` lines (R8).
- Option B: drop the accumulation clause; the norm is the tolerance alone, stated in the spec as a function of the reduction length K (RA R8, RA4).
- Option C: `|r - exact| <= n * u * sum|terms|` plus one rounding, with n the term count and u the unit roundoff of the result type; restricted to these three words; `reduce_max` and `reduce_min` stay exact (DB6).
- Option D: the oracle accumulates wider and rounds once to the result type, so goldens stay bit-exact; narrower hardware accumulation falls outside the norm (oracle review-response, R8 objection; critic single-rounding point).
- Evidence: DB6; SC24; RA R8, RA4 (owner of the tolerance: spec, `taidl.testing`, or `conftest.py:227` `ULPS`), RA5 (GPU TF32 versus HIGHEST precision on `dot`); holes.md H6, H7; dynamics.md numeric classes; spec-draft S8.4 "Float accuracy"; `examples/mini_trn/mini_trn.py:91`; oracle `README.md:163`; oracle O13, O14, O22.
- Binding: OD16; OD14 (StableHLO leaves accumulation open, so the norm cannot be inherited).

### Q1.6: Is each float operation rounded once to its result type, and are denormals kept or flushed?

- Context: DB7, DB8.
  The float exact class needs one rounding per operation, but XLA's default `xla_allow_excess_precision=true` lets bf16 and f16 chains stay at f32; no plan item sets the flag.
  H6 and dynamics.md say "denormals kept"; no test backs it, and the XLA CPU client or the target hardware may flush.
- Option A (rounding): every float operation rounds to its result type once; the oracle sets `--xla_allow_excess_precision=false` before jax initializes, with a test on a bf16 chain (DB7).
- Option B (rounding): excess precision is allowed inside the tolerance class; the exact class is limited to operations where it cannot occur.
- Option C (denormals): keep denormals (H6, dynamics.md).
- Option D (denormals): flush every float operation's denormal input and result to a zero of the same sign (DB8).
- Option E (denormals): fix the choice only after a test on the pinned jaxlib, then state one norm and never "either" (DB8).
- Evidence: DB7; DB8; critic.md:43; holes.md H6; dynamics.md numeric classes; `mini_trn.py:101-104` (bf16 softmax chain); oracle O12 (`decorator.py:23`), O22.
- Binding: OD16; R6 (pinned jax version).

### Q1.7: How are NaN, signed zero, and special values of approximate words defined?

- Context: DB14, DB16, SC24.
  dynamics.md says NaN propagates as a quiet NaN and leaves sign and payload open, while `bitcast_convert` exposes them.
  The sign of zero from `maximum` and `minimum` on (-0, +0) is not stated.
  A per-word tolerance does not pin `exp(-inf)`, `log(0)`, or `tanh(inf)`.
- Option A (NaN): any NaN equals any NaN; payload and sign are not part of the meaning; `bitcast_convert` of a NaN is therefore the one non-bitwise point of the exact class (DB14).
- Option B (NaN): one canonical NaN bit pattern per type is the result of every NaN-producing operation.
- Option C (zero): `maximum` and `minimum` follow IEEE 754-2019: NaN propagates and -0 < +0 (DB14).
- Option D (zero): follow whatever the pinned XLA does, stated per word after test.
- Option E (special values): special values of every approximate word are exact (inf, zero, NaN results match the IEEE function) (DB16).
- Option F (special values): special values fall inside the per-word tolerance; NaN and infinity compare by a stated rule (SC24).
- Evidence: DB14; DB16; SC24; dynamics.md numeric classes; spec-draft S8.4; oracle test bounds (`utils.py:584-612`).
- Binding: OD16.

### Q1.8: What does `convert` do out of range?

- Context: DB9.
  No ruling covers `convert` out of range; the spec draft S8.4 gives float-to-integer saturation and NaN to 0 and leaves other pairs open.
  dynamics.md puts `convert` in the float exact class with nearest-even rounding, which is wrong for float to int.
  StableHLO marks out-of-range conversion TBD.
- Option A: float to float and int to float: nearest even, overflow to signed infinity, or NaN where the type has no infinity (`f8e4m3fn`); float to int: truncate toward zero, saturate, NaN to 0; int to int: wrap; to `pred`: nonzero is true (DB9).
- Option B: float to int out of range or NaN is a fault.
- Option C: float to float overflow saturates to the largest finite value (the rule for `f8e4m3fn` generalized).
- Option D: int to int is a fault when the value does not fit the target.
- Each of A to D combines per conversion class; the panel may rule per class.
- Evidence: DB9; SC9 (sub-byte, I to F, `f8e4m3fn` no infinity); spec-draft S1.3.12, S8.4 integer edge cases; oracle coverage-table (float `convert` overflow row, O22); holes.md H6.
- Binding: OD16; OD14 (StableHLO silent, XLA value is today's behavior).
- Note: a saturating ISA writes `clamp` before `convert` under option A.

### Q1.9: Does tensor data ever fault? Integer divide by zero, shifts, and index clamping

- Context: DB10, SC9, SC10, H17.
  An expression `/` or `%` by a computed 0 faults (S1.3.2), while integer `divide` and `remainder` by 0 inside a semantics return fixed values (spec-draft S8.4).
  `dynamic_slice`, `dynamic_update_slice`, and `gather` take indices from tensor data; StableHLO clamps them and neither S1.3 nor dynamics.md says so.
  The oracle checks all addressing before tracing (`utils.py:651-806`); a data-dependent fault needs a device-side flag and commit-after-check (O23; R55 objection on cost).
- Option A: tensor data never faults; in-semantics indices clamp as StableHLO does; integer `divide` and `remainder` by 0, `min / -1`, `min % -1`, and wide shifts take fixed values (XLA's, per S8.4); faults depend only on attributes, state, and region text (DB10).
- Option B: integer `divide` and `remainder` by 0 and `min / -1` fault like the expression `/`, one norm for both; indices still clamp (SC10, first alternative).
- Option C: out-of-range indices in `dynamic_slice`, `dynamic_update_slice`, and `gather` also fault (SC9 alternative).
- Option D: two norms stay deliberate: expressions fault, tensor words return fixed values (SC10, second alternative).
- Evidence: DB10; DB12; SC9; SC10; RA6 (adopt H16, H17); holes.md H5, H17; spec-draft S6.2, S8.4, S8.1; oracle fidelity.md O23; critic.md:46; oracle review-response (R55 commit-after-check).
- Binding: OD16 ("one simple norm easy to explain").
- Cost point to weigh: options B and C require the O23 flag and commit-after-check in the oracle and a defined result for the same case in the backend.

### Q1.10: What may a literal be, and how is a float literal rounded?

- Context: R57, DB15, SC17, H15, O8.
  The generators disagree today (`spec_checks.py:94-107`, `utils.py:398-426`); the oracle rounds float literals twice (via f64), and an integer literal above 2^53 at a float type also rounds twice.
  R57: the value must fit the element type, floats round once to nearest even, out-of-range literals are rejected, out-of-range backtick expressions fault.
  "Fit" is undefined for a float that rounds to infinity or underflows.
- Option A: a finite decimal that rounds to infinity (NaN at `f8e4m3fn`) is rejected; underflow rounds normally subject to Q1.6 (DB15).
- Option B: a literal rounds by IEEE rules from the exact decimal, so overflow yields infinity and is accepted.
- Option C: any literal that does not round-trip exactly is rejected.
- Option D (separate axis): a backtick constant outside its element type faults at issue (S1.3.9, R57); alternative is rejection of the issue before any run.
- Evidence: DB15; SC17 (expression literal range is 0 <= k < 2^63; only an attribute can be -2^63); H15; R57; oracle O8 (`Fraction` rounding, hex bits); spec-draft S6.2, S7.5, S1.3.9.
- Binding: OD16.

### Q1.11: What is the start value of `reduce_<reducer>` and the sign of a float sum over zeros?

- Context: R3, DB13, SC22, H7.
  R3 and the spec draft S8.4 give the identity of the reducer at the type: 0 for add, 1 for multiply, least and greatest for max and min, with `f8e4m3fn` using its largest finite value; a float sum starts at +0.0.
  A float sum over a size-1 dimension is then not the element itself for -0.0, since -0.0 + +0.0 = +0.0.
- Option A: the identity as the oracle has it (`reduce_init`, `utils.py:439-451`), a float sum from +0.0, an all-negative-zero sum is +0.0 (R3, DB13, matches XLA).
- Option B: the float sum starts from -0.0, the additive identity under IEEE 754, so a sum of negative zeros is -0.0.
- Option C: an explicit `init` operand per reduce line (rejected in semantics/rulings.md; the grammar lacks the form).
- Evidence: R3; DB13; SC22; holes.md H7; semantics/rulings.md "Reduction initial values"; spec-draft S8.4 initial value table; `utils.py:439-451`; reduction order tied to Q1.5.
- Binding: OD16.

### Q1.12: What does a fault guarantee, and what does a producer owe?

- Context: R55, R4, R10, DB5, DB11, RA4.
  R55: a fault writes no region and no state, and the generator reports it with the clause id.
  The oracle review-response says this needs commit-after-check for data-dependent errors, a cost the ruling does not size.
  R4 and R10 say the backend "may assume" a violation never happens; H11 and owner decision 16 phrase it as a producer that never emits one.
  dynamics.md says a REQUIRE failure inside a constraint "does not hold", while R55 calls it a fault.
  The backend chooses `@a.` values and placements, so for bounds and constraints it is a producer, not an assumer; R21 has no backend plan item.
- Option A (atomicity): a fault writes no region and no state, for every fault including data-dependent ones (R55).
- Option B (atomicity): a fault writes nothing only for faults the host checks before any write; the program stops at a fault and writes before it are not defined as rolled back.
- Option C (constraint failure): any REQUIRE failure, in a constraint or elsewhere, is a fault; the backend's reading is "never selected" (DB11).
- Option D (constraint failure): a REQUIRE failure inside a constraint means the constraint does not hold, which is a fault only through S1.3.5 (dynamics.md, H4, H5).
- Option E (producer contract): the backend never emits an issue that faults, and lists each precondition it cannot see (state before R27, `|d0|`) in its coverage doc (DB5, RA R4, R10).
- Option F (producer contract): the backend may assume preconditions hold (R4, R10 as drafted).
- Evidence: DB5; DB11; RA R4, R10, R21; SC1 (update fault precedes constraint test); SC6 (fault versus reject vocabulary); holes.md H5, H11; spec-draft S1.3, S5.6, S6.3; oracle review-response "Objections" on R55; oracle fidelity.md O23.
- Binding: OD16; R44 clause ids.
- Dependency: option A is cheap if Q1.9 option A holds and costly if Q1.9 option B or C holds.

## For the authors

Minor findings that need no debate; each is a correction or an alignment to a ruling already made.

- DB12: add `abs` and `negate` of the minimum wrapping, `clamp` as `min(max(x, lo), hi)`, shift amount read unsigned, integer `dot` exact with wraparound, integer `power` per XLA; verify each on the pinned XLA, and align the S8.4 negative-exponent sentence with the result.
- DB17: R7's reason cites placement-mark code; cite `instruction_codegen.py:732-737` in act-oracle.
- DB18: R4 covers a size below 1; replace the S1.3 undefined rows with the fault list and a pointer to the numerics norm table; empty S1.3.15 once Q1.5 is decided.
- DB19: delete the H3 remedy of rejecting looped starts (R28 unrolls every loop); make O11 conditional on the R28 spike.
- DB5, R21: add a backend plan item for allocator equality next to B11; add a backend item for Q1.4 once decided.
- SC5: number every REQUIRE of dynamics.md (including reshape element count H10, backtick dimension at least 1, constraint failure) so clause ids attach.
- SC6: use "fault" for issue-time run-time errors and "reject" for front-end errors only.
- SC7: rename "tensor buffer" and "control register" to buffer and state in the semantics track (OD19).
- SC12: instruction without semantics has no regions at all; a semantics implies at least one output (R11).
- SC13: define initial `Mem`, `Reg`, and `|d0|` in the machine state.
- SC14 to SC16: re-pin stale citations (grammar, `names.py`, spec-draft, backend, oracle lines) or cite section ids.
- SC17, SC18: split the literal range of expressions from attributes; operands are `%` names except `parameter(k)` and `constant(lit)`.
- SC19, SC20, SC21: mark WF-IO as front end only when regions are given; state that `compare` `type`, when given, lies in the allowed set; check `broadcast` and `dynamic_slice` side conditions against the pinned StableHLO.
- SC22: say "the element combined with the identity" for size-1 and empty reductions.
- SC23: reword "total by construction" and "implemented identically in both generators" in semantics/review.md; division by `(0)` or `-0` is a fault, not a rejection.
- SC25, SC26: rename semantics plan items to avoid the F1 to F23 collision; pick one owner for the `|d0|` source (issue-time parameter or declared size).
- Spec draft S5.5 still describes `set_apart` and `set_no_partial_overlap` that R58 removes; S1.3.6 and S5.3 follow Q1.1 and Q1.3.
- Output regions that overlap are a fault in every source (S1.3.4, H2, RA3); the oracle must stop letting the later write win.
- Align S6.2 "a literal `0` divisor is rejected, a computed 0 is a fault" with R55 vocabulary.
