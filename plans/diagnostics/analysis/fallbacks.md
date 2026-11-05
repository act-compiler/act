# Silent fallbacks where an error belongs

| Where | Fallback | Effect | Instead |
| --- | --- | --- | --- |
| `act-backend/act_backend/generator.py:119-121` | An instruction with no semantics is left out of the backend | The compiler never selects it; no message | Reject at the front end when an instruction has regions and no semantics (B24); if it is allowed, list it as unsupported-by-design |
| `act-oracle/taidl_to/semantic_generator.py:19` | An instruction with no semantics and no output regions generates an empty body | The oracle runs it as a no-op | Same rule as above |
| `taidl/taidl/instruction.py:90, :210-215` | `check_values` runs only when both `set_inputs`/`set_outputs` and `add_semantics` ran | A semantics with `%In1` and no `set_inputs` passes the front end | Check once at the end of the description |
| `taidl/taidl/instruction.py:103-110` | A second `set_inputs` replaces the regions and drops placement marks | A mark vanishes without a message | Reject the second call, or keep and say so in S5.2 |
| `taidl/taidl/accelerator.py:40-43` | `base_dir` falls back to `os.getcwd()` | Generated files land in the working directory | Keep; document in S4.2 as implementation-defined |
| `taidl/taidl/instruction.py:112-129` | Region starts and sizes are checked for length only | Size 0 and negative literals reach generators; S5.2 and S1.3 call them undefined | Reject literal sizes below 1 |
| `act-backend/act_backend/spec_checks.py:394-415` | A value's element type is never compared with its data model's | The backend emits code for a mismatched description that the oracle rejects (`semantic_generator.py:45`) | Reject in the front end (U3, U10) |
| `act-backend/act_backend/generic/src/isel/extractor/fast.rs:191` | `debug_assert!(false)` compiled out; the loop `break`s | A search with no progress ends with no error | Return `Err` |
| `isel/initializer/parser.rs:310-314` | With no ROOT line, the last line is the root | Matches XLA text; defined | Keep; state it in the backend's input rules |
| `isel/initializer/parser.rs:268-306` | A line that is neither `ENTRY` nor a region is skipped | A stray line in the kernel is ignored | Reject a line outside any computation |
| `generic/src/isel/initializer/line.rs:90-92` | Blank lines and `//` comments are skipped | Defined; harmless | Keep |
| `act-oracle/taidl_to/generic/utils.py:268-271` | A value past signed 64 bits emits a `UserWarning` | Execution continues; S1.3 calls the case implementation-defined | Raise a `V` error under the v3 norm (no warning) |
| `act-oracle/taidl_to/generic/shapes.py:77-146` | Helpers return `None` for unknown shapes and the check is skipped | A line shaped by expressions is checked at the call only | Keep; name the two-phase check in the oracle docs |
| `act-backend/act_backend/generic/src/pipeline.rs:266` | A dropped candidate is printed on stdout | The reason is lost unless every candidate fails | Keep the first-drop summary (`:208-218`); add a count |
| `act-backend/act_backend/spec_checks.py:22-27` | Only instructions with semantics are checked (`generator.py:119-121`) | An unselectable instruction is never validated | Validate every instruction, then skip |
| `act/act/generators/__init__.py:6, :13` | A missing generator package surfaces as a bare `ImportError` | No pointer to the install step | Catch at the entry and raise an `X` row naming the package |
