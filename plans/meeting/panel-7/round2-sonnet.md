# Panel 7, round 2: Sonnet (lens: new user)

Rebuttals after reading round1-opus.md and round1-fable.md.
Owner decisions bind; the coordinator's ruling that the prose term for `d0` is "hbm" (Decision 19) is applied in Q7.7.

### Q7.1

All three agree: Option A, one new v3 tutorial, R54-b; the layout of the new tutorial is decided inside that item.
- Strongest opposing argument (Opus): add Option C's layout (`tutorials/events/<event>.md`, past trees on tags) so master holds one copy.
- Rebuttal: Option C is compatible with A and costs nothing extra once archived trees leave master.
- Evidence: both Opus and Fable cite 57 files with non-ASCII under tutorials, so R54-b avoids a rule change (engineering.md ASCII rule).
- Final position: Option A, R54-b; event pages as Opus describes.
- Confidence: high.
- Still divides: nothing material; Fable defers layout, Opus fixes it now.

### Q7.2

All three agree: Option A, containing C (words-to-avoid generated from the banned column).
- Final position: Option A with the generated words-to-avoid.md.
- Confidence: high.
- Still divides: nothing.

### Q7.3

- Strongest opposing argument (Opus): "matching generator releases" in Decision 10 reads most simply as the same major, so taidl 3.0, taidl-to 3.0, act-backend 3.0, act 3.0, and one number names one concept.
- Rebuttal: act-backend is at 1.16 and act pins it by tag (pyproject.toml:15-19), so 3.0 skips 2.x with no release behind it.
- Rebuttal: the compatibility table in the README is the map either way, and check_versions.py enforces it.
- Concession: Opus's reading of "matching" is plausible, and Decision 10 does not settle it; Fable also flags that an owner statement would decide.
- Evidence: README.md:4 already says "act-oracle v3"; taidl-to is 2.11, so 3.0 is the next major there anyway.
- Final position: Option A with the A23 item; numbers taidl 3.0, act-oracle 3.0, act 3.0, act-backend 2.0 as default; the numbering rule goes to the owner as a value judgment.
- Confidence: high on A23, low on the numbers.
- Still divides: act-backend 2.0 (Sonnet, Fable) against 3.0 (Opus).

### Q7.4

- Strongest opposing argument (Fable, Option A): the README's first command is `pip install git+...@v2.4` and CI never runs it, so the advertised path is untested; a gate per tag costs one CI job.
- Rebuttal to my round 1: I said a plain-host install needs Rust, ANTLR, and OR-Tools.
- Evidence against that: the generated parser files are checked in (taidl/taidl/antlr4/IDLV2ExprLexer.py), taidl depends only on `antlr4-python3-runtime==4.9.2` (taidl/pyproject.toml:15), and act-backend depends only on taidl (act-backend/pyproject.toml:15-17).
- Evidence: cargo, CMake, and OR-Tools are needed only when `generate_backend` runs (act-backend/README.md:34), not at install.
- So the install itself needs Python 3.14, git, and wheels (act-oracle pins jax 0.11.2 and numpy, act-oracle/pyproject.toml:15-21); a stock runner with setup-python can do that.
- Concession: AG15 and my round 1 overstated the install cost; a fresh-venv gate for install, import, and the oracle quickstart is cheap and tests the exact first command a newcomer runs.
- Residual: backend generation on a plain host needs OR-Tools and nlohmann headers, so that half stays Docker-only.
- Final position: Option A scoped to install plus oracle generation in a fresh venv as a release gate; Docker is the documented path for backend generation; no wheels in v3.
- Confidence: medium.
- Still divides: Opus holds Docker-only (B); the panel splits on whether the gate covers backend generation.

### Q7.5

All three agree: Option A, a complete quickstart run by A20; the tutorial link waits for the v3 tutorial; A3 keeps only the newest line that check_versions.py requires.
- Difference: Fable sets a 15-25 line bound, I size it by need; Fable and I both allow moving it to docs/ if honest code exceeds about 25-30 lines.
- Final position: Option A.
- Confidence: high.
- Still divides: only the length bound.

### Q7.6

- Strongest opposing argument (Opus): the code already says "state variable" for one entry, so the term is in use, not new.
- Evidence verified: `@s.` is labeled 'state variable' beside 'addressing attribute' and 'computational attribute' (taidl/taidl/instruction.py:397), and accelerator.py:47 says "machine state variable".
- Fable's objection that "variable" reads as mutable program state fails: state is written by instructions (`update`), so "variable" is accurate here.
- My round-1 collision argument (backend "tensor variables", `var_name`) is weak: `var_type` is renamed `element_type` under Q7.10, which removes the collision in the API.
- Concession: unban "state variable" as the singular; "state entry" has no source in code or the papers.
- Names: `add_buffer(name, access_dims, base_element_dims, element_type)` and `add_state(name, value)` (all three agree; Opus cites PyTorch `register_buffer`).
- Definition: "a named value of a declared kind, an integer by default" (R14, AG8).
- Final position: Option A.
- Confidence: medium.
- Still divides: Fable keeps "a state" or "state entry".

### Q7.7

- Owner ruling applies: the prose term for `d0` is "hbm" and "global memory" is not an option, so my round-1 hybrid is withdrawn.
- Strongest opposing argument (Opus, Fable): one spelling for identifier and prose removes a mapping step.
- Final position: Option A; "hbm" is the built-in buffer `d0`, sized by the `hbm` kernel argument; ban "global memory", "off-chip memory", "DRAM", "main memory"; drop every HBM rename in A4.
- GPU scope: device memory is `hbm`; shared memory and register fragments are buffers (all three agree).
- Confidence: high (binding).
- Still divides: nothing; the glossary must still name the alias relation to the paper word once, in the row, as a pointer and not as a synonym.

### Q7.8

All three agree on "fault", "reject" only at generation and kernel compile time, and deleting "undefined behavior" and "implementation-defined".
- Strongest opposing argument (Fable, on scope): rename "refus" in prose and new code only; the bulk rename and `test_refusals.py` belong to a separate test-suite item.
- Strongest opposing argument (Opus, Option B rebuttal): coverage status is another concept, so "implementation-defined" must not be reused; coverage says "unsupported".
- Rebuttal to Fable: user-visible messages and public names carry the banned word otherwise; a docstring such as "a refused or failed generation" (act-oracle/taidl_to/generator.py:25) is read by users.
- Concession to Fable: the bulk rename is a plan item of its own, not a glossary edit, and the item list should say so.
- Evidence: instruction.py:453 contains "refused" (verified), so the AG11 correction should add :453 to the cites, not remove it.
- Final position: Option A; rename in messages, public names, and test file names in the v3 batch, scheduled as its own item.
- Confidence: high on the words, medium-high on scope.
- Still divides: prose-only (Fable) against full rename (Sonnet, Opus).

### Q7.9

- Strongest opposing argument (Opus, Fable): whether `set_inplace` exists is a language question for the semantics panels; a vocabulary row that picks a side pre-empts R5 and RA2.
- Rebuttal attempted: Decision 17 names an "in-place mark", which presupposes a mark.
- Concession: Decision 17 fixes the meaning ("the output reuses the marked input's storage") and does not fix the spelling; R5 stays open in the audit (RA2), and this panel is vocabulary only.
- Evidence: mini_trn.py:97-99 repeats region text in input and output, which supports text identity once but does not settle equal regions spelled differently.
- Final position: Option C; define the row by Decision 17 now, mark the identifier column pending R5 and RA2, delete the placement-mark row now (R58).
- Confidence: medium.
- Still divides: nothing among debaters; the pending row resolves when the semantics panels rule.

### Q7.10

- Strongest opposing argument (Fable, Opus): v3 is the only window under Decision 10; deferring grammar names leaves two names for one construct, and the grammar listing in the spec shows them.
- Fable's claim that rule names leak into ANTLR error text is half right: the listener stores only `msg` (instruction.py:23-29), so rule names do not appear, but token names can appear in "expecting" sets, and `OPERATION` is a composite token (IDLV2.g4:45).
- Stronger argument: the grammar rule `instruction` (IDLV2.g4:8) names a line of the semantics language while `Instruction` (instruction.py:33) names an ISA instruction, so one document would use "instruction" for two concepts.
- Concession: rename the grammar rule and token too; the cost is generated parser churn, paid once in the batched release.
- Keep `update`, `update_asts`, `update_text`; drop the five control-register renames (AG5; all three agree).
- Sub-question: generated per-repo table of old name, new name, consumer files, citing symbols, plus a CI grep that old names return nothing (all three agree).
- Final position: Option A.
- Confidence: medium-high.
- Still divides: nothing.

### Q7.11

All three agree: Option A; keep repo, distribution, module, and function names; keep `act-ecosystem` and fix the sentence (R52); `targets/` becomes `accelerators/` in the v3 rename table.
- Strongest opposing argument (Option D, exempt `targets/`): users' scripts read the generated path.
- Rebuttal: Decision 10 batches breaking changes, and the path is listed in the rename table and the migration note.
- Final position: Option A with the `targets/` rename.
- Confidence: high on prose names, medium on the path.
- Still divides: nothing; evidence of external scripts reading `targets/` would flip the path.

### Q7.12

All three agree: Option A (error names the replacement and the clause id, R44) plus a migration note generated from the Q7.10 table; no converter.
- Difference: I proposed stub errors for renamed v2 methods; Opus and Fable say no aliases.
- Resolution: a stub that only raises is not an alias, because it keeps no working second name; Fable's "use X instead" message is the same mechanism.
- Evidence: Python deprecation practice (PEP 387) assumes warnings across releases, which a single declared break does not use (Opus).
- Final position: Option A plus the generated note; a raising stub per renamed method is allowed, a working alias is not.
- Confidence: medium-high.
- Still divides: whether stubs are worth a line per rename.

## Escalation to the owner

None.
No counter-evidence to an owner decision is strong enough to raise.
