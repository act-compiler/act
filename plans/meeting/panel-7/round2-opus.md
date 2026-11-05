# Panel 7 round 2: Opus (docs and terminology owner)

Binding inputs: owner decisions 6, 10, 14, 16, 17, 19, 20, 21, plus the owner ruling that the prose term for `d0` is "hbm" (decision 19).

### Q7.1

- All three agree on Option A with R54-b; the layout (Option C) is the only split.
- Strongest opposing argument (Fable): Options C and D decide the layout of trees about to be archived, so the layout belongs inside the new tutorial item.
- Concession: Fable is right; the layout matters only for the one v3 tutorial, and deciding it now pre-empts that item.
- Final position: Option A with R54-b; the layout is decided inside the v3 tutorial item.
- Confidence: high.
- What still divides the panel: nothing of substance.

### Q7.2

- All three agree on Option A containing Option C: terms.md is the only glossary, words-to-avoid.md is generated from its banned column, and superseded files are deleted.
- Strongest opposing argument (Sonnet): the glossary must say why the papers differ, since the papers say "tensor buffers" and "control registers".
- Rebuttal: the reason is one source-column cite ("Decision 19") per overridden row, not a prose note, so the row stays a statement of what is.
- Final position: Option A with the Option C generation; overridden rows cite the owner decision in the source column.
- Confidence: high.
- What still divides the panel: nothing.

### Q7.3

- All three agree on the A23 item and on merging A16 and A17 into S8 and A23.
- Strongest opposing argument (Fable, Sonnet): act-backend is at 1.16, so 3.0 skips a major that never existed; the rule is "each repo bumps its own major in the same train".
- Rebuttal: skipping a number to align versions has precedent, since Angular went from 2 to 4 so the core matched the router's 3.x line.
- Concession: my step 5 claimed the compatibility table becomes a check, but check_versions.py keeps the table for every v1 and v2 row anyway, so the saving is smaller than I said.
- Sonnet's README.md:4 cite ("an even faster simulator in `act-oracle` v3") fits both readings, since act-oracle reaches 3.0 either way.
- The evidence is balanced and the choice is a value judgment about what "matching" means in decision 10.
- Final position: Option A23; numbers go to the owner as two options: act-backend 3.0 (one major across four repos) or 2.0 (own major bumped).
- Confidence: high on the item; low on the numbers.
- What still divides the panel: act-backend 3.0 against 2.0.

### Q7.4

- Strongest opposing argument (Fable): the README's first install command is the `pip install git+...@v2.4` path, and CI never runs it, so the advertised path is the untested one.
- Rebuttal: the fix is to stop advertising it as the supported path, not to add a toolchain gate.
- A plain-host install needs cargo, CMake, make, a C++17 compiler, OR-Tools, and nlohmann/json at generation time (act-backend README.md:30), so a stock-runner gate rebuilds the image's toolchain by hand.
- Sonnet shows that the image build already runs `pip install /opt/act`, which resolves the git-pinned siblings (act docker/Dockerfile.amd64:120-127), so the pin resolution is exercised.
- Sonnet also shows that wheels (Option D) cannot remove the Rust need, because generating a backend builds it with cargo (act-backend README.md:30).
- Adopt Sonnet's gate: the release runs the Q7.5 quickstart inside the built image.
- Final position: Option B; Docker first in the README, plain-host requirements listed as unsupported, release gate runs the quickstart in the image.
- Confidence: medium.
- What still divides the panel: Fable keeps Option A; the deciding fact is who the users are, and no one has cited evidence on it.

### Q7.5

- All three agree on Option A, and on A3 keeping only the newest "moves the ecosystem" line that check_versions.py requires.
- Strongest opposing argument (Sonnet): size the snippet by need rather than at 15 lines; the end-to-end test accelerator is 37 lines (act tests/test_end_to_end.py:30-66).
- Concession: agreed; the length is set by the smallest accelerator that runs load, compute, and store, with a linked example file if it passes about 30 lines.
- Final position: Option A, runnable and run by A20, linking the spec and coverage tables; the tutorial link waits for the v3 tutorial (decision 6).
- Confidence: high.
- What still divides the panel: nothing.

### Q7.6

- All three agree on `add_buffer(name, access_dims, base_element_dims, element_type)` and `add_state(name, value)`, and on defining state to cover declared kinds (R14).
- Strongest opposing argument (Sonnet, Fable): "variable" collides with tensor variables and `var_name` and `var_type`, and reads as mutable program state to an MLIR reader.
- Rebuttal: "state variable" is already the user-visible term in code, messages, and docs, not a proposal.
- The error text names it: `KINDS = {..., '@s.': 'state variable'}` (taidl/taidl/instruction.py:397), giving "'@s.busy' names no state variable" (taidl/tests/test_expressions.py:229).
- The docs use it: taidl/docs/expressions.md:3,5,13,16 and taidl/docs/instructions.md:13.
- The docstring uses it: "Declare a machine state variable" (taidl/taidl/accelerator.py:47).
- The collision weakens in v3: `var_type` becomes `element_type` (Q7.10), and `var_name` sits on the class that becomes `Buffer`.
- State is mutable: an instruction's `update` lines assign it, as in `@s.<name> = 1` (taidl/docs/instructions.md:13), so "variable" is accurate, not misleading.
- "State entry" has no source in code, papers, or docs, so Option B coins a term and forces a rename of every message above.
- Final position: Option A, singular "state variable"; banning it would rename shipped messages for a coined word.
- Confidence: medium; the singular is a naming choice the owner can overrule.
- What still divides the panel: the singular; Sonnet and Fable keep "state entry" or "a state".

### Q7.7

- The owner ruled that the prose term for `d0` is "hbm" (decision 19), which settles the question; "global memory" and other alternatives are not options.
- All three now land on Option A; Sonnet's B/D hybrid falls under the ruling.
- Banned words: "global memory", "off-chip memory", "DRAM", "main memory", and "HBM" as a name for `d0`.
- GPU sub-question, where all three agree: device memory is `hbm`, and shared memory and register fragments are buffers.
- Final position: Option A under the owner ruling, with the GPU mapping above.
- Confidence: high.
- What still divides the panel: nothing.

### Q7.8

- All three agree on the words: "fault" at run time, "reject" at generation and kernel compile time, and "undefined behavior", "UB", and "implementation-defined" banned.
- Strongest opposing argument (Fable): the bulk rename of "refus" is a test-suite item, not a glossary item, so rename it in prose and new code and keep `test_refusals.py` until then.
- Fable also corrects the cite: taidl/taidl/instruction.py:453 reads "one bare string is refused", so the word is at :116, :145, :451, and :453.
- Concession: the cite is right; the line list in my round 1 missed :453.
- Concession: test file names are not user vocabulary, so `test_refusals.py` can move in whichever test item renames tests.
- Rebuttal kept: messages and public identifiers are user-visible, so they rename in v3; a prose-only rename leaves users reading the banned word.
- Final position: Option A; rename "refus" in messages, docstrings, and public identifiers in v3; test file names follow the test item.
- Confidence: high on the words; medium on the rename scope.
- What still divides the panel: whether user-visible messages rename in v3 (Sonnet and I) or only in new code (Fable).

### Q7.9

- Strongest opposing argument (Sonnet): decision 17 reads "the output reuses the marked input's storage", which presupposes a mark, so R5's removal of `set_inplace` conflicts with a binding decision.
- Concession: the argument decides the question; decision 17's topic is "In-place mark" and its text names a "marked input".
- A derivation from identical region texts marks nothing, so R5 as drafted cannot meet decision 17's text.
- My round-1 step 3 ("if R5 stands, nothing is marked") is exactly why R5 cannot stand under a binding decision; I drew the wrong conclusion from it.
- Option C left open a question the owner already closed; the semantics panels may set the mark's spelling, not whether it exists.
- Sonnet's edge evidence holds: the mark is checked as one flag per input (taidl/taidl/instruction.py:163-170) and used in mini_trn softmax (act-backend/examples/mini_trn/mini_trn.py:97-99).
- MLIR bufferization takes the same meaning: an in-place op reuses the buffer of its destination operand.
- Final position: Option A (changed from C); row "in-place mark": `set_inplace`, the output reuses the marked input's storage; delete the placement-mark row (R58).
- Confidence: medium-high.
- What still divides the panel: Fable keeps Option C pending R5 and RA2; that holds only if decision 17 can be met with no mark, and its text says otherwise.

### Q7.10

- All three agree on renaming the public API names and keeping `update`, `update_asts`, and `update_text`, with a grep-generated per-repo table citing symbols and a CI grep for old names.
- Strongest opposing argument (Sonnet): grammar rule names never appear in an ISA description, so renaming them gives a newcomer nothing.
- Concession on rules: parse-tree rule names reach only code, in taidl/taidl/operations.py, act-oracle/taidl_to/oracle_visitor.py, act-backend/act_backend/spec_checks.py, and act-backend/act_backend/isel_ir2isa_rewrites_txt_generator.py.
- Those consumers already call the rule's node `line` (taidl/taidl/instruction.py:293, taidl/taidl/operations.py:175), so the code has its own name for it and the spec can state the term.
- Rebuttal on tokens: the syntax-error listener passes ANTLR's message through unchanged (taidl/taidl/instruction.py:29-30).
- ANTLR's default messages list expected tokens by display name, and `OPERATION` has no literal (taidl/taidl/antlr4/IDLV2.g4:45), so its symbolic name can reach users; I did not reproduce this, since antlr4 is not installed on the host.
- Fable's step 2 claim that rule names leak into error text is not supported by the listener; only token names can.
- Final position: rename the public API names (Option A list) and the token; keep grammar rule names (Option B for rules).
- Confidence: medium.
- What still divides the panel: Fable renames the rules too; Sonnet keeps the token; a captured syntax-error message settles the token.

### Q7.11

- All three agree on Option A: prose "test oracle generator" and "compiler backend generator"; repo, distribution, module, and function names kept; one glossary row per name; bare "kernel" only for `@kernel`.
- All three keep `act-ecosystem` (R52) and rename `targets/<name>/` to `accelerators/<name>/` in the v3 batch, each at low or medium confidence.
- Final position: Option A, `act-ecosystem` kept, `targets/` renamed in the v3 rename table.
- Confidence: high on prose; medium on the path.
- What still divides the panel: nothing; the path rename waits only on evidence of external scripts that read `targets/`.

### Q7.12

- All three agree on Option A plus a migration note generated from the Q7.10 table, with no converter and no working aliases.
- Strongest opposing argument (Sonnet): a renamed v2 method should raise a stub error naming its replacement, since otherwise the user meets a bare `AttributeError`.
- Concession: adopted; a stub that only raises is not an alias, because it admits no v2 call, so one name per concept holds.
- The stub list and the note come from the same Q7.10 table, so they cannot disagree.
- Final position: Option A plus raising stubs for renamed methods, both generated from the rename table.
- Confidence: high.
- What still divides the panel: nothing.

## Escalation to the owner

- None; no owner decision conflicts with the evidence above.
- Q7.3 goes to the owner as a value judgment, not an escalation: act-backend 3.0 or 2.0.
