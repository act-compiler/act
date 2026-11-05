# Panel 11 round 2: Opus

Lens: language designer and front-end maintainer.
Code paths are relative to /home/user; plan paths are relative to /home/user/act/plans.
The panel agrees on option A for all four questions; this round settles the spellings and the two real splits.

### Q11.1

Strongest opposing argument (no debater held it, so I restate the case for B fairly): a standalone text can be read and checked without running the author's code, and a non-Python consumer such as the fsim-lib direction (decision 23) could parse it directly (Sonnet's mind-changer, round1-sonnet.md:28).

Rebuttal:
1. Today's specification is already a Python file the front end executes (taidl/examples/mini_trn.py:1-27), so tracing adds no new trust boundary.
2. A non-Python consumer is served by serializing the checked `Model`, panel 3's second-consumer trigger, not by a second surface language; Sonnet reaches the same conclusion (round1-sonnet.md:28).
3. Fable's mind-changer, chained comparisons `a < b < c` (round1-fable.md:23), is closed by the tracer: Python expands a chain to `and`, which calls `__bool__` on a symbolic value, and `__bool__` raises at the author's line (docs.python.org/3/reference/expressions.html, comparisons).
4. One hazard none of us priced: on tensor values Python `/` reads as true division, while StableHLO `divide` truncates integers toward zero and `remainder` takes the dividend's sign (openxla.org/stablehlo/spec, divide and remainder).
5. So the tensor overloads are `+`, `-`, `*`, and unary `-` only; `ops.divide` and `ops.remainder` are called by name, and `/`, `//`, `%` on a tensor raise naming them.
6. Sonnet's list "`+ - * / %` map to the S8 table" (round1-sonnet.md:85) should be narrowed this way; on integer attributes `//` and `%` keep S6's floor meaning exactly (spec-draft/expressions.md:28-29).
7. Sonnet's HLO text printer (round1-sonnet.md:13) is worth keeping as a debug print of the `Model`, never as an input form.

Final position: option A by tracing, no source parsing, zero third-party dependencies.
Confidence: high.
What still divides the panel: nothing on the option; the tensor operator set of step 5 is new and unreviewed.

### Q11.2

Strongest opposing argument (Fable): `buffer[start : start + size]` is the spelling every NKI example uses, so an architect reads it with no new API, and Python checks arity at trace time (round1-fable.md:30, :45).

Rebuttal on regions:
1. A Python slice is start and stop, while a TAIDL region is start and size (taidl/taidl/instruction.py:96-113); the slice forces the author to write the start twice (`addr_in : addr_in + n * 128`).
2. The tracer must then recover the size by symbolic subtraction of two expressions, and a stop not of the form `start + e` (for example `0 : k` against a start of `1024 - a`) has no size at all without an algebra the front end does not own.
3. Fable's own partial-unit write needs a chained sub-view, `psum[0:m, bank : bank + 1][:, 0, 0:n]` (round1-fable.md:61, :64), which is a second region construct with its own bounds rule, and he rates its spelling medium.
4. Sonnet's region object is the better half of my own form: `r = acc.region([1024 - a], size=[64])` names a region once and is both read and written, which is exactly the accumulate case (round1-sonnet.md:76-78 writes it twice for lack of that).
5. I keep the keyword `size=` so no reader takes the second argument for a stop.

Concessions, each with its reason:
1. Constraints use `&` and `|` on conditions (Fable, Sonnet), dropping my `any_of`: S6 has no bitwise operator on integers (spec-draft/expressions.md:25, :32), so `&` and `|` are unambiguous on a condition and raise on an integer.
2. The precedence trap `acc == 0 | acc == 1` parses as the chain `acc == (0 | acc) == 1`, which reaches `__bool__` and raises, so it never runs silently.
3. A marker is a keyword on `write` naming the read value, `write(r, v, inplace=old)` (round1-sonnet.md:39, :78), not my `mark=` on the read: it pairs one input with one output by object, so it survives a future multi-output instruction, where today's flag lists cannot (instruction.py:163-231).
4. Fable's `inplace=['bank_in']` on the decorator names a parameter that his signature lacks (round1-fable.md:55-57), which shows why a string name is the worse spelling.
5. The encoding layer leaves the body: `mini_trn.encode(gemm_acc, word=32, fixed=..., fields=...)` (round1-sonnet.md:81-82), since decision 38 adds it "on top of" layer 0 and S11.3 says the oracle and backend never read it (spec-draft/layers.md:32-46).
6. Its fields are keyed by parameter name, which `check()` matches against the signature; naming a parameter is not inferring meaning from a name (decision 34).
7. Buffers take keywords, `buffer('d1', access=[128], unit=[64], dtype=bf16)` (Fable, Sonnet), over my positional lists.
8. The width annotation is a subscript, `Addr['u16']`, following both others; a string key avoids 126 type objects for `s1` to `s64` and `u1` to `u63` (spec-draft/layers.md:22-24).

Rebuttal on state: Fable's `trn.rows = ...` (round1-fable.md:68) makes a later read of `trn.rows` in the same body return the new value by Python's meaning, which contradicts S5.4's parallel assignment; `update(ctr, ctr + 1)` with every read giving the pre-issue value (round1-sonnet.md:37) has no such reading.

Final position: typed parameters; `buffer.region(starts, size=)` objects with `read(r)` and `write(r, v, ...)`; markers as write keywords naming read values; `require(...)` with `&` and `|`; `update(...)`; width as `Addr['u16']`; encoding as a separate `encode(instr, ...)`; `meta=` on accelerator, buffer, and instruction; result types inferred except where an operation needs a target.
Confidence: medium-high on regions, markers, and state; medium on the remaining spellings.
What still divides the panel: slice versus region object (Fable); in-body versus separate encoding is now closed on Sonnet's form.

### Q11.3

Strongest opposing argument (Fable, Sonnet): S1.2 says a generator rejects a specification using a feature it does not support, naming the feature (spec-draft/overview.md:26), and decision 42 names the backend's machine-state requirements as checks, so a backend-unsupported instruction should reject, not drop (round1-fable.md:85; round1-sonnet.md:107).

Rebuttal and partial concession:
1. Decision 40 rejects only problems the architect can correct, and says a backend drops a failing candidate with a recorded reason (minutes/owner-decisions.md:34).
2. An instruction with two outputs or an address-dependent size is a valid ISA; the architect cannot "correct" the hardware, so rejecting the whole specification fails decision 40.
3. S1.2 itself has a `partial` status, "works with stated limits" (spec-draft/overview.md:28), which is the drop-with-reason behavior listed in the backend's docs.
4. I concede the split: a machine-state requirement the architect can meet (state declared, initialized, integer) rejects with code and line, as Sonnet lists (round1-sonnet.md:107); a valid instruction beyond the backend's model is dropped and recorded; an empty usable set rejects.
5. I concede Sonnet's convenience: a generator entry given a raw `Accelerator` calls `check()` itself, once, so `Model` stays the only type a generator body reads (round1-sonnet.md:100).
6. Fable's form for extra checks, plain functions `(model) -> list[Rejection]` run after the shared check (round1-fable.md:81), is what I meant by `x_checks(model)`; no registry, all three agree.
7. Sonnet's `python -m taidl check spec.py` (round1-sonnet.md:103) is the standalone use decision 42 asks for, and I adopt it.

Final position: option A with the type barrier; collect every error with code, section id, and the author's file and line; backend rejects correctable machine-state faults and drops unsupported valid instructions with a reason; oracle adds none.
Confidence: high on the barrier; medium on the reject-versus-drop split.
What still divides the panel: whether a backend-unsupported operation word rejects (Fable, Sonnet) or drops its instruction (me).

### Q11.4

Strongest opposing argument (Fable): panel 3 ruled "no converter" because decision 11 leaves no capacity (meeting/panel-3/decision.md:172-181), so the converter is a development script on the corpus, never shipped (round1-fable.md:98).

Rebuttal and partial concession:
1. Panel 3's reason was the cost of writing a converter; here the corpus forces one anyway, since 25 Python files in taidl, act-oracle, and act-backend call `add_semantics` (grep this session).
2. Once written against v2 taidl, putting it in a final 2.x release is packaging, not capacity; I keep that as the recommendation but not a requirement.
3. The v3 removed-names table catches `add_semantics` with one coded rejection naming the converter, which is panel 3's mechanism unchanged (meeting/panel-3/decision.md:175-177).

On order of work, where Fable and Sonnet put the tracer first and the generator port last (round1-fable.md:99; round1-sonnet.md:125-133):
1. Porting every generator to read only `Model` while the string front end still exists keeps each port testable on unchanged inputs; Sonnet names that port the critical path (round1-sonnet.md:133), which argues for starting it first.
2. The tree-to-`Model` builder this needs is not throwaway: it is the converter's front half and the equivalence oracle, string-built `Model` equal to traced `Model` up to call sites.
3. That direct comparison beats printing HLO text and reparsing with the 2.x parser (round1-sonnet.md:123), which tests the printer as much as the surface.
4. Sonnet's item 1, the operation table as data (round1-sonnet.md:126), belongs at step 1 beside the node model, and I adopt it.

Final position: option A; order is node model and operation table, `check()` from the string front end with every generator ported to `Model`, traced surface, converter on the corpus, `Model` equivalence per specification, then delete ANTLR in v3.
Confidence: medium-high.
What still divides the panel: shipping the converter in 2.x versus a repository script (Fable), and generators-first versus tracer-first.
