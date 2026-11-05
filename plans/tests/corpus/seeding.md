# Seeding the corpus

T7 moves the existing front-end cases into the corpus one outline file at a time, each written in the authoring surface.
A case seeded from a v2 string-form file is converted by the converter script and gated by the equivalence test of T37; a case with no ISA specification behind it is written by hand.
T8 turns the oracle's reference tests into the operations corpus with a stated reference and tolerance per case.

## T7. Seed the corpus from existing front-end tests (S2 to S7)

- What: turn each parametrized case of `taidl/tests/test_grammar.py` (:57-120 strictness, :146-186 root, :194 inputs, :352-360 literals, :405 shapes) and the rejection cases of `test_instruction.py` and `test_expressions.py` into corpus directories in the authoring surface, then delete the originals that the corpus covers; a lexer or grammar assertion on the string form has no v3 successor and becomes a `trace` or `check` case of T38, or is dropped with its justification.
- What: `act-backend/tests/test_refusals.py:208-358` and `act-oracle/tests/test_refusals.py:87-130` hold ISA specifications that a generator rejects but the front end accepts; they enter the corpus as `accepted` cases with generator expectations in T12.
- What: `act-oracle/tests/test_kernel.py:1287-1728` (looped attributes, region shapes at the call) enters as S5.2 cases with kernels.
- What: tests of removed constructs become `rejected` cases whose code names the replacement: the linalg words (`act-oracle/tests/test_linalg.py:1`); they are seeded after the v3 API is fixed.
- What: the marker and overlap tests (`test_instruction.py:269-440`, `test_refusals.py:383-396`) seed the T30 matrix, since the markers stay as shorthand for constraints.
- What: a parsed-value assertion such as `test_grammar.py:357` becomes a `model` table entry, so no fact is lost.
- Benefits: the biggest existing body of language tests becomes tagged and shared; the two generator rejection files shrink to expectation rows.
- Caveats: the grammar tests exercise the lexer directly (`IDLV2Lexer` at `test_grammar.py:9`); the v3 front end has no lexer, so a token-level fact is kept only where the surface reproduces it, and those tests are deleted with the ANTLR grammar in the last step of the order of work (Q11.4).
- Fail points: a case that passes for the wrong reason, as with the two assertion-free tests at `test_instruction.py:50` and :222; the runner asserts the verdict, so a silent pass is impossible.
- Order and dependencies: after T1, T5, T6, and T37; several PRs, one outline file per PR (declarations, instructions, expressions, semantics).
- Verification: each PR deletes the parametrize block it replaced, and T28 finds no test id gone without a mapped successor; S2 to S7 each have at least one accepted and one rejected case.
- Breaking: no.
- Files: `taidl/tests/test_grammar.py`, rejection cases of `test_instruction.py` and `test_expressions.py`, ISA specifications of both `test_refusals.py`, `act-oracle/tests/test_kernel.py:1287-1728`; every case preserved as a corpus directory.

## T8. Operations corpus with reference values (S8)

- What: one case per tensor operator and element type the language specification lists in S8.3, with an expected result stored in the case, following `act-oracle/tests/test_elementwise.py:406` and `test_reduce.py:87-129`.
- What: v3 regenerates the stored values from NumPy (pinned version named in `case.toml`) or from the closed form S8.4 gives; no StableHLO interpreter is added.
- What: the tolerance per case is `expect = "exact"` or `"almost"`; a float `dot` or `reduce` case states the S1.3.15 result-type bound from its term count and result type, other float cases state `ulps`.
- What: the language specification owns the bound, `taidl.testing` implements it, and the conftest imports it.
- What: NaN, signed-zero, and infinity comparison is stated once in `taidl.testing`; a case may set `nan = "any"`.
- What: S8.5 (operations outside the language) gives `rejected` cases, one per word the language leaves out; the surface has no such `ops.<word>`, so the case expects the `trace` error that names the supported table.
- What: one `fault` case per run-time fault the norm defines, as `program.toml` programs (T29), seeded from `act-oracle/tests/test_kernel.py:590` (region outside its buffer), :712 and :749 (pieces outside hbm, overlapping), :1229 and :440 (zero divisor), and `act-backend/tests/test_refusals.py:351-358`.
- What: a fault case depends only on attributes, state, and region text, so no fault case reads tensor data.
- What: a seed whose divisor is a computed expression stays a `fault` case (S1.3.2); a seed whose divisor is tensor data becomes a fixed-value case.
- What: a fault case expects the oracle to raise the code with buffers and state unchanged, and a fixed-value case expects the stored result with no raise; the backend never faults, and T13 checks that simulating its assembly never does.
- What: fixed-value cases for S1.3.12 and S1.3.17, each with a stored expected value and `reference` naming the XLA rule or NumPy:
  - integer `x / 0` is -1 (signed) or all ones (unsigned), `min / -1` is `min`, and `x % 0` is `x`;
  - float-to-integer `convert` truncates toward zero, saturates out of range, and maps NaN to 0;
  - a divisor of 0 in a padded lane of a masked `divide` returns the fixed value and writes no fault.
- What: the overlap cases are T30; seeds `act-backend/tests/test_extraction.py:441-563` and `test_constructs.py:448` enter there.
- Benefits: the oracle's reference tests become language tests any generator can run; the backend runs the same kernels through T13.
- Caveats: the removed words `cholesky` and `triangular_solve` (`test_linalg.py:94-190`) become `rejected` cases (T7); the words that stay keep a NumPy reference with a stated tolerance.
- Fail points: a reference computed with the library the oracle lowers to (JAX) proves nothing; `case.toml` names the reference library and T8 requires it is not JAX.
- Order and dependencies: after T5; needs L3 for the S8.3 and S8.4 tables; lands in parallel with T7; the S8 word and type cases land before T11.
- Verification: every row of S8.3 has a case; every case names a reference and a tolerance.
- Breaking: no.
- Files: the six act-oracle reference files and `test_templates.py`; every case and tolerance preserved as corpus data, the files stay as runners; `act-oracle/tests/conftest.py:226-227` (`ULPS`) is replaced by the bound from `taidl.testing`.
