# Test bugs

Defects in the tests themselves, found by reading; none was run.
Severity: high means the test can pass while the behaviour it names is broken; medium means it fails on harmless changes; low is hygiene.
Each fix sketch names the plan item that carries it.

| ID | Severity | Where (file:line) | Bug | Fix sketch |
| --- | --- | --- | --- | --- |
| TB1 | high | `taidl/tests/test_instruction.py:50` | `test_identifier_accepted` has no assertion; passes by not raising | assert the name is stored on the instruction (T1) |
| TB2 | high | `taidl/tests/test_instruction.py:222` | `test_one_start_and_size_per_addressed_dimension` has no assertion | assert the region shape recorded (T1) |
| TB3 | high | `act-backend/tests/test_refusals.py:363-421` | five tests (`test_sizes_the_allocator_reads`, `test_constraints_the_allocator_holds`, `test_placement_mark_where_a_size_divides_is_held`, `test_placement_mark_that_appends_no_constraint_passes`, `test_state_no_instruction_uses_passes`) assert nothing; they only check that generation does not raise | become `supported` rows in T11 whose runner asserts a tree exists (T12); the two placement-mark tests become T31 rows that assert the allocation |
| TB4 | high | `act-backend/tests/test_package.py:17` | `test_wheel_builds_with_the_least_setuptools` asserts nothing beyond `check=True` | assert the wheel file exists and names the version |
| TB5 | -- | `act-oracle/tests/test_templates.py:510` | retired: `test_long_operand_list_laid_out` asserts through `check_oracle`, which runs ruff format and lint | none; it is a layout test for T4 |
| TB6 | medium | `act-oracle/tests/test_constant_iota.py:107-157` and `test_linalg.py:139` | eight tests assert only inside conftest helpers (`each_type`, `assert_exact`); a helper change silences all of them at once | keep, but T8 cases carry their own expected value and tolerance so the helper cannot pass vacuously |
| TB7 | medium | `act-backend/tests/test_matching.py:648-661` | selection asserted by parsing `api.` calls from assembly text; a correct alternative selection fails | compare results through the oracle (T13); keep selection counts only in impl tests |
| TB8 | medium | `act-backend/tests/test_mini_trn.py:49-56` | exact call counts of the attention kernel pinned; a cheaper correct selection fails | drop the count, keep the reference and golden checks (T14) |
| TB9 | medium | `act-backend/tests/test_mini_trn.py:72`, `test_mini_trn_double.py:74` | pins generated Rust text `TensorOp::InstrSoftmax(..) => &[true],` | replace with the T31 tied row, which asserts the shared storage on the assembly, then delete (T4) |
| TB10 | medium | `act-backend/tests/test_generators.py:239`, :264 | pins generated Rust text `TensorOp::InstrSum30(..)` and `InstrVadd` | assert through `cargo check` of the generated file or delete (T4) |
| TB11 | medium | `act-backend/tests/test_generators.py:175-393` | nine tests re-run rustfmt and clang-format layout that `conftest.py:468-472` already enforces on every generated tree | delete (T4); the mark test at :243 stays |
| TB12 | medium | `act-backend/tests/test_refusals.py:164-176`, :200-205 | regexes over `egraph.rs` and `dtype.rs` source; a refactor of the match arms fails them while the behaviour holds | the invariant (generator words equal reader words) becomes a T11 row set checked by T12 compiling each word |
| TB13 | medium | `act-backend/tests/test_constructs.py:500` | counts `'dtype': jnp.bool_` occurrences in the kernel text | run the kernel and check the pred result (T13) |
| TB14 | medium | `act-backend/tests/conftest.py:463-476`, :487-495 | `work_dir`, `assembly`, `output` read `request.module.EXAMPLE_DIR`; a module without the constant fails at setup with an attribute error | explicit `example` fixture parameter (T2) |
| TB15 | medium | `act-backend/tests/conftest.py:468-472` | a formatter complaint inside the `work_dir` fixture fails every test of the module at setup, not one named test | move to `test_generated_sources_are_formatted` (T2) |
| TB16 | medium | `act-oracle/tests/test_mini_trn.py:20` | `generate_and_import_oracle` runs at import time; a generation error is a collection error for the whole session | session fixture (T2) |
| TB17 | medium | `act-oracle/tests/conftest.py:75-155` | `LOWERED` is a hand copy of the generator's table; `test_refusals.py:44` fails on drift without saying which side is right | one expectations table rendered to coverage.md (T11) |
| TB18 | medium | `act-backend/tests/test_mini_trn.py:40-43` and `act-oracle/tests/test_mini_trn.py:22-35` | the same golden is checked in two repos with two `regenerate.py`; a regeneration in one leaves the other stale | one golden and one script in act (T14, T15) |
| TB19 | low | `act-oracle/tests/conftest.py:227` | `ULPS` tolerance table is undocumented policy and one value per type, so it cannot bound a K-term sum | tolerance per case from the language specification's bound, implemented in `taidl.testing` (T8) |
| TB20 | low | `taidl/tests/test_grammar.py:317-350` | `TestDocs` parses markdown tables by regex and fails on a column reorder | render the tables from the operations module (T11 pattern) or keep as doc-sync with a stated table shape |
| TB21 | low | `taidl/tests/test_examples.py:92` | pins `IDLV2Parser.ModuleContext` as the parsed type | assert the semantics parsed through the public API (T10) |
| TB22 | low | `act-oracle/tests/test_mini_trn.py:22` | GPU skipped wholesale for goldens; the reference check at :38 would still run but is not separated from the skip | GPU checks are manual and use the reference tolerance (T15) |
