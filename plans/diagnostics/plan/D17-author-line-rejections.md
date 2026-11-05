# D17 Rejections at the author's line

- What: the tracer raises a coded `R` error, in the architect's own frame, for each spelling the surface refuses: `RF058` (integer operators), `RF059` (tensor operators), `RF060` (a slice for a region).
- What: each message names the spelling found and the spelling to use, for example `/` and `//`, or `ops.divide` for a tensor `/` (U12, scheme.md wording rules).
- What: `__bool__` on a symbolic value raises once for `if`, `and`, `or`, and a chained comparison, so one test covers them.
- What: `check` adds the collected entries `RF061` (two updates of one variable) and `RF062` (a marker or an encoding field key naming nothing).
- What: line 2 of each entry is the author's file and line; a trace-time error also appears in the Python traceback at that line.
- What: a trace-time error stops at once and is not collected; a `check` entry is collected with the others (Q11.3).
- Benefits: the error line is the architect's own line; no position is counted inside a string (today's `text_error`, taidl/taidl/instruction.py:379-383).
- Caveats: the spelling rows are seeded with the surface; a spelling renamed before v3 renames its message, never its code.
- Fail points: a spelling that takes a silent Python meaning; noticed by one test per row asserting code, line, and the spelling named.
- Order and dependencies: D1, D2, U12.
- Verification: a test per `RF058` to `RF062` row asserts the code, the file and line of the faulty call, and the replacement named in the message.
- Breaking: yes.
