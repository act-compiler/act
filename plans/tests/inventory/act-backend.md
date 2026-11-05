# Inventory: act-backend

The test files of the act-backend repo, 223 tests, with a verdict each.
Style values are those of the inventory key.

| file | what it tests | style | verdict | reason |
| --- | --- | --- | --- | --- |
| `tests/test_matching.py` (34, 1479 lines) | which instruction is selected for which kernel, on bespoke ISA specifications | pr, impl | reshape: split into T13 differential cases and a trimmed impl file | each test carries its own ISA specification and kernel text; selection facts such as :898 `dot` contracting pairs and :1247 slice ranges are spec-visible, while :661 reads `log/*.py` and :1002 reads assembly text |
| `tests/test_constructs.py` (21) | mini_trn with constructs added, one per PR | pr | merge into T7 and T13 | :500 counts `'dtype': jnp.bool_` occurrences in the kernel text; :433 and :454 parse `api.` calls by regex |
| `tests/test_endings.py` (26) | how a compile ends: exit status, output files, log directories | cli | keep | contract of the binary, not of the language; :620-641 constant tests belong in T13 |
| `tests/test_reader.py` (16) | HLO kernel reader: what it reads, what it rejects | cli, spec | keep, tag | the kernel is HLO, not TAIDL, so this is the backend's own input contract; :240-298 sum and dot reading overlap T13 |
| `tests/test_extraction.py` (14) | e-graph extraction candidates | impl | keep, trim | reads `.pii` log lines at :87, :135, :481, :598; keep as impl tests, never as conformance |
| `tests/test_generators.py` (30) | generator functions without a build; formatter layout of long names | impl | delete the layout tests, keep the rest | :175-393 pin rustfmt and clang-format layout that `conftest.py:468-472` already checks on every generated tree, and rustfmt runs at generation; :239 and :264 pin `TensorOp::InstrSum30(..)` text; the mark test at :243 stays |
| `tests/test_refusals.py` (28) | ISA specifications the generator rejects, each by its error | spec, impl | reshape into T11 and T12 | :149-157 and :185-197 walk grammar words and types and are the seed of coverage.md; :164-176 and :200-205 regex `egraph.rs` and `dtype.rs`; :363-421 hold no assertion |
| `tests/test_generation.py` (18) | build environment: solver paths, cargo, lock file | pkg | keep | environment contract; :277-338 layout tests overlap `test_generators.py` and go with them |
| `tests/test_allocator.py` (5) | allocator drops a candidate with a reason | impl | keep | C++ allocator contract on hand-made candidates |
| `tests/test_cost.py` (11) | the `act_cost` C library contract | cli | keep | out of the language's scope |
| `tests/test_names.py` (2) | every name an ISA specification gives compiles in the generated sources | spec | keep, extend in T9 | property-shaped already at :95 |
| `tests/test_package.py` (2) | the wheel's contents and build | pkg | keep | :17 holds no assertion and passes when the build does not raise |
| `tests/test_mini_trn.py` (8), `tests/test_mini_trn_double.py` (8) | generate, compile, simulate the attention kernel; golden | e2e, golden, impl | merge into T14 | :49-56 pin exact call counts, :72 pins `TensorOp::InstrSoftmax(..) => &[true],`; the golden check at :40-43 duplicates the oracle's |
| `tests/conftest.py` (496 lines) | fixtures and helpers | - | reshape (T2, T3) | `work_dir` at :463-476 and `plain_dir` at :480-484 read `request.module.EXAMPLE_DIR`; formatter assertion inside the fixture at :468-472 |
| `tests/lint.py`, `tests/lint.sh` | clippy on a generated crate and the compiler on its allocator, then again with a cost library | script | keep | run by CI in the test image; not tests |
| `tests/regenerate.py`, `tests/regenerate.sh` | rewrite an example's golden from its compiled kernel | script | keep | the record format at `regenerate.py:35-52` stays (T15) |
| `tests/cost/model.c`, `no_cost.c`, `fail.c` | cost libraries that `test_cost.py` builds and loads | fixture | keep | inputs of `test_cost.py:39-141`; out of the language's scope |
| `tests/pytest.sh` | runs the suite in the image | script | keep | the entry point of CI |

## Counts

| keep | reshape | merge | delete |
| --- | --- | --- | --- |
| 112 | 62 | 37 | 12 |

- Delete is the layout tests of `test_generators.py` and the three text or regex pins at `test_refusals.py:164-205` and `test_mini_trn*.py:68-80`, as counted on the planning day.
- `test_refusals.py` is renamed `test_rejections.py` for the tests that remain after T12.
