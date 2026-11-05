# Failure policy corpus

T36 turns owner decision 40 into cases: nothing fails without a genuine cause.

## T36. Failure policy cases (decision 40)

- What: three groups of rows, one per tool; each row names the ending it expects: a result, a recorded drop, or a stop with a `code`.
- What: backend candidate rows: a corpus kernel with an injected candidate failure completes, emits assembly, exits 0, and the drop log holds the candidate id, a `code`, and a reason.
- What: an internal error inside candidate search (an `I`-coded failure injected through a test hook) is a recorded drop with its `IB` code, never a stop; the run summary counts it.
- What: a run with a dropped candidate yields the same assembly as a run where that candidate never existed, in deterministic mode (T27).
- What: a kernel where every candidate drops ends with the aggregated no-assembly error that lists the drop reasons by code; the error names the instruction the ISA specification lacks, which the architect fixes.
- What: an I-coded drop on a corpus kernel outside an allowlist fails the test run as a finding, not the user's run.
- What: input-kernel rows: malformed kernel text and a kernel outside the input specification each stop with a `K` code naming the kernel line (`kernel.hlo:12`) and the construct to change.
- What: one input-kernel row per format as each reader lands: StableHLO, XLA HLO, torch-mlir, and torch.fx; a format with no reader yet has a `pending` row.
- What: oracle rows: invalid assembly (unknown mnemonic, wrong operand count, attribute outside its type) stops with a `K` code naming the assembly line; a binary word matching no instruction stops with the decode error naming the word offset.
- What: oracle rows: a valid program on every corpus ISA specification runs to the end, including the fixed-value cases (divide by 0 and out-of-range `convert` in tensor data), with no stop.
- What: a fault that a valid-looking program reaches (S1.3) is the program's violation, reported as a `V` code at the issue; no other oracle ending exists.
- What: exclusion rows (Q11.3): an instruction the backend cannot model (state read or update, two outputs, a size over an addressing attribute, a constraint the allocator cannot hold, a tensor operator outside the table) is excluded with `code`, clause, and author line; the ISA specification generates, exits 0, and the report is printed and stored in the crate.
- What: an exclusion is never a stop: a kernel that avoids the excluded instruction compiles, and a kernel that needs it is rejected at kernel compile time naming it, as a coded `K` row.
- What: the oracle excludes nothing and registers no check of its own; a valid ISA specification with a state-updating instruction runs on it.
- What: front-end and generator rows: every `rejected` corpus case, trace exception or `check` entry, carries a message naming the object and the fix (`expected` and `found` per `scheme.md`), and its class is `R` or `U`, never `I`.
- What: a sweep runs both generators over every accepted corpus case; each ends in a result or in its expectations row (`unsupported` with a code), never in an `I`, `X`, or uncoded stop.
- Benefits: a stop without a cause (a panic in search, a generator crash on a valid ISA specification) fails a named row.
- Caveats: the injection hook exists only in test builds; the drop log is read from stderr and the drop file of D16.
- Fail points: a drop that hides a wrong answer; the same-assembly row and the T13 simulation of every emitted assembly rule it out.
- Order and dependencies: after T12, T13, T29, T27, and D16; Q2.
- Verification: each ending class above has a row; the sweep count equals the accepted case count.
- Breaking: no.
- Files: new cases under `taidl/conformance/` and `act-backend/tests/test_endings.py` (extended, from D7); seeds `pipeline.rs:265-266` and `:59-60` behavior.
