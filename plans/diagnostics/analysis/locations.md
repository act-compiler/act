# Messages that do not name the offending instruction or source location

## Front end

- No message names the ISA specification file and line; only the Python traceback does, and only for errors raised directly in a user call.
- A rejection raised in a generator (`spec_checks.py`) has no traceback into the ISA specification at all, since the instruction object keeps no call site.
- `taidl/taidl/names.py:10, :12, :14` name the kind and the name, not the owner: an attribute name error names the instruction only through `kind` (`instruction.py:67`).
- `taidl/taidl/accelerator.py:41, :43, :56` name the data model or state variable; `:50` names the state variable and not the instruction that came first.
- `taidl/taidl/instruction.py:166, :169` name the instruction but not which input.
- A text fault names a line and column counted in the text (`instruction.py:345-368`); a backtick expression adds the semantics offset, but a start or size error says `start '{text}' line 1, column C` with no region index.

## act-backend

- Python rows name the instruction (`spec_checks.py:26-415`) but not the ISA specification location.
- `spec_checks.py:302` names the data model; `:292, :294` name `{name}: its operand on '{buffer}'` with no region index.
- Rust kernel errors name the value (`%x`) but not the kernel line number: `isel/initializer/lib.rs:154-187`, `nodes.rs:39-431`, `expression.rs:106-464`, `line.rs:96`.
- `pipeline.rs:59-60` reports one aggregated reason for a whole search (`every candidate was dropped, the first because ...`); the other dropped candidates are only on stdout at `:266`.
- `generic/src/ir/*.rs` panics carry no instruction or value name (`pii.rs:36-48`, `egraph.rs:407` names shapes only).
- C++ rows name a tensor (`parser.cc:152-363`) but the wrapper adds the candidate line text, not the kernel line (`act_malloc.cc:62-63`); `emitter.cc:30` is a fragment (`a constant of element type {t}`) that only reads well after the wrapper at `:317`.
- `constraint.cc:130-187` and `parser.cc:13` print to `cerr` and assert; they name no tensor.

## act-oracle

- Generation rows name the instruction (`oracle_visitor.py`, `semantic_generator.py`).
- `generic/utils.py:555` (`copy of {x} : {t} does not have the result type {t2}`) names no instruction.
- `generic/utils.py:675-692` (`debug: ...`) and `decorator.py:21, :150` (`unknown simulation platform`, `unknown oracle mode`) name no kernel.
- `generic/utils.py:80` raises `ZeroDivisionError(env)` with the iteration environment only; the rewrap at `templates/templates.txt:320, :360` adds the instruction.
- Call-time rows add `at i = v` for the loop iteration (`utils.py:662-783`), which is the model to copy.

## act

- `act/tests/check_versions.py:96-99` raise a bare `TypeError` (`None` is not subscriptable) when a pattern is missing, naming no file.
- `act/docker/run.sh:176-182` and `:138-143` print the same prompt text twice, once per code path.
