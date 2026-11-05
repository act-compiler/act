# Candidate extraction

Every few rewrite iterations, `pass` (`generic/src/pipeline.rs:226-262`) runs both extractors and allocates each candidate.
The fast extractor yields one candidate graph, each e-class once per data model (`isel/extractor/fast.rs:12`); the slow one enumerates repeat-free graphs up to a node limit that grows each pass (`slow.rs:13`, `pipeline.rs:501-502`).
Both repeat the same constant detection (`slow.rs:21-23`, `fast.rs:19-21`); the cost is known only after allocation, which is why extraction enumerates instead of using an egg `CostFunction`.

| ID | Item | Depends on | Size (S/M/L) | Quarter (Q1/Q2) |
| --- | --- | --- | --- | --- |
| B10 | One extractor module with two strategies over shared code | B1, B3 | M | Q1 |
| B11 | In-place copies as a candidate rewrite, not a fix-up | B10 | M | Q2 |

## B10 One extractor module with two strategies

- What: one `extraction` module; constant detection, data model resolution and candidate building shared; the fast and the slow extractor are two enumeration functions over it.
- Preserves: both strategies, the node limits, candidate order and `--log` numbering.
- Benefits: removes duplicated passes between `slow.rs` (283 lines) and `fast.rs` (208 lines); names say what each does.
- Caveats: the candidate order feeds the numbering of `--log` files and tie-breaking; keep it.
- Fail points: a changed candidate order changes which equal-cost candidate wins; noticed by goldens.
- Order and dependencies: after B1, B3; benefits from B9.
- Verification: goldens byte-identical; same candidate count per pass in `--verbose`.

## B11 In-place copies as a candidate rewrite

- What: `fix_inplace` (`isel/extractor/inplace_fix.rs:9`) edits each candidate after extraction to copy an overwritten shared input; express it as a step of candidate building with its cost visible.
- Preserves: the copy rule of `inplace_fix.rs` and its two error cases.
- Benefits: one place decides copies; the fix-up's two error returns become rejections with reasons at one site.
- Caveats: needs a copy instruction per data model that holds a shared overwritten input.
- Fail points: a candidate that overwrites a live value; noticed only by oracle simulation, so add a kernel test with a shared overwritten input.
- Order and dependencies: after B10; T adds the kernel test.
- Verification: goldens byte-identical; new kernel test simulates correctly under the oracle.
