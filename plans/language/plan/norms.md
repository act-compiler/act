# Plan: defined-behavior norms and removals

These items state the run-time norms of panel 1 in the spec and remove the constructs panel 2 cuts.
Sizes assume the shared 3-4 engineers; every item marked breaking ships together in the TAIDL v3 release with a migration note.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| L32 | Defined-behavior norms: pre-state issue, unwritten reads, numerics, fault guarantees | L1, L2 | M | Q1 |
| L33 | Remove rare words and redundant spellings, with one coded rejection | L1 | S | Q1 |

## L32 Defined-behavior norms

- What: S1.3, S5.4, S5.6, S7.5, and S8.4 state the panel 1 norms, one sentence each.
  - Every expression of an issue reads the pre-state; updates and outputs commit together, and updates are a parallel assignment (Q1.1).
  - A read of an on-chip element no earlier issue wrote is a fault; `d0` holds the caller's `hbm` (Q1.4).
  - One rounding per float operation, denormals kept, NaN and signed zero per IEEE 754-2019, `bitcast_convert` of NaN gives the canonical quiet NaN (Q1.6, Q1.7).
  - `convert` out of range per conversion class (Q1.8); tensor data never faults, and indices clamp (Q1.9).
  - A float literal rounds once from its exact decimal, and one rounding to an infinity is rejected (Q1.10); a float sum starts from +0.0 (Q1.11).
  - A fault writes nothing, the program stops at the first fault, every REQUIRE failure is a fault, and B never emits a faulting issue (Q1.12).
  - Program order is the only order; a constraint used as a wait faults when false (Q2.6).
  - An attribute value may be negative; implicit region constraints apply to the evaluated address only (owner decision 34).
- Changes: the spec pages above; O moves update emission after the outputs, tracks a written mask per buffer, and sets `xla_allow_excess_precision`, `xla_cpu_ftz`, and `xla_cpu_enable_fast_min_max` false; B lists the preconditions it cannot see.
- Breaking: yes, a kernel that read an unwritten buffer or relied on post-state starts changes.
- Benefits: every run-time case has one norm, so generators can be compared verdict by verdict.
- Caveats: the denormal sentence is final only after a per-word test on the pinned jaxlib passes on both platforms; a stale value in a reused buffer is not detected.
- Fail points: an oracle compile option reverts; notice by T tests of a bf16 chain, `maximum(-0, +0)` in both orders, and `f32` to `f8e4m3fn` overflow.
- Order and dependencies: after L1 and L2; O and B own their code changes.
- Verification: each norm has a T test; S1.3 has no open row.

## L33 Remove rare words and redundant spellings

- What: remove `cholesky`, `triangular_solve`, and `batch_norm_inference` (Q2.8), presumed reshape, and `add_constraints` beside `constraints=` (Q2.9, D34).
- Also: a semantics defines at least `%Out0`, and an instruction omits semantics only when it has no regions.
- Changes: `taidl/operations.py`, `instruction.py`, and the removed-names table that raises one coded `RF` rejection naming the construct, its replacement, and the section id (Q3.12).
- Breaking: yes, one release with a migration note.
- Benefits: the `BOOL`, `TRANSPOSES`, and `NUMBER` attribute forms go with the words; one spelling per construct.
- Caveats: the marks stay (owner decision 30) and are not in the table; the table is seeded once the v3 API is fixed; no converter.
- Fail points: a removed construct raises a bare `AttributeError`; notice by a test per table row.
- Order and dependencies: after L1; L30's renames feed the same table.
- Verification: each table row has a test asserting `code` and `object`.
