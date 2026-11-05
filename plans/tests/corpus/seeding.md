# Seeding the corpus

T7 moves the existing front-end cases into the corpus one outline file at a time.
T8 turns the oracle's reference tests into the operations corpus with a stated reference and tolerance per case.

## T7. Seed the corpus from existing front-end tests (S2 to S7)

- What: turn each parametrized case of `taidl/tests/test_grammar.py` (:57-120 strictness, :146-186 root, :194 inputs, :352-360 literals, :405 shapes) and the rejection cases of `test_instruction.py` and `test_expressions.py` into corpus directories, then delete the originals that the corpus covers.
- What: `act-backend/tests/test_refusals.py:208-358` and `act-oracle/tests/test_refusals.py:87-130` hold ISA descriptions that a generator rejects but the front end accepts; they enter the corpus as `accepted` cases with generator expectations in T12.
- What: `act-oracle/tests/test_kernel.py:1287-1728` (looped attributes, region shapes at the call) enters as S5.2 cases with kernels.
- Benefits: the biggest existing body of language tests becomes spec-tagged and shared; the two generator rejection files shrink to expectation rows.
- Caveats: the grammar tests exercise the lexer directly (`IDLV2Lexer` at `test_grammar.py:9`); a case that needs a raw token stream carries `semantics.txt` and is run by the lexer path of T6.
- Fail points: a case that passes for the wrong reason, as with the two assertion-free tests at `test_instruction.py:50` and :222; the runner asserts the verdict so a silent pass is impossible.
- Order and dependencies: after T1, T5, T6; several PRs, one spec file of the outline per PR (grammar, declarations, instructions, expressions, semantics).
- Verification: each PR deletes the parametrize block it replaced; S2 to S7 each have at least one accepted and one rejected case.
- Breaking: no.
- Files: `taidl/tests/test_grammar.py`, rejection cases of `test_instruction.py` and `test_expressions.py`, spec texts of both `test_refusals.py`, `act-oracle/tests/test_kernel.py:1287-1728`; every case preserved as a corpus directory.

## T8. Operations corpus with reference values (S8)

- What: one case per operation word and element type the spec lists in S8.3, with a reference result computed by NumPy or by the closed form S8.4 gives, following `act-oracle/tests/test_elementwise.py:406` and `test_reduce.py:87-129`.
- What: the tolerance per type moves from `act-oracle/tests/conftest.py:227` (`ULPS`) into the case file so the spec, not a conftest, states it.
- What: S8.5 (operations outside the language) gives `rejected` cases, one per word the grammar lexes and the language leaves out.
- What: one `error` case per run-time violation the norm defines, seeded from `act-oracle/tests/test_kernel.py:590` (region outside its buffer), :712 and :749 (pieces outside HBM, overlapping), :1229 and :440 (zero divisor), and `act-backend/tests/test_refusals.py:351-358`; the oracle must raise the defined error and T13 checks the backend agrees.
- What: S5.5 in-place cases where a kernel reads the marked input after the instruction and must see the output, since an in-place mark means the output reuses the input's storage; seeded from `act-backend/tests/test_extraction.py:441-563` and `test_constructs.py:448`.
- Benefits: the oracle's reference tests become language tests any generator can run; the backend runs the same cases through T13.
- Caveats: linalg words (`test_linalg.py:94-190`) have no closed form and keep a NumPy reference with a stated tolerance.
- Fail points: a reference computed with the same library the oracle lowers to (JAX) proves nothing; the case file names the reference library and T8 requires it is not JAX.
- Order and dependencies: after T5; needs L's S8.3 and S8.4 tables; lands in parallel with T7.
- Verification: every row of S8.3 has a case; every case names a reference library and a tolerance.
- Breaking: no.
- Files: the six act-oracle reference files and `test_templates.py`; every case and tolerance preserved as corpus data, the files stay as runners.
