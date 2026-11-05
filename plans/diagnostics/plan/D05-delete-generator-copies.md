# D5 Delete duplicated checks in generators

- What: remove from `act_backend/spec_checks.py` the rows that D4 moved (`:75, :195, :210, :218, :292, :294, :302, :415`) and from the oracle the matching rows (`semantic_generator.py:16, :40, :45, :53`, `oracle_visitor.py:255, :261, :289, :387`).
- What: a generator that still meets an `R` condition raises `I` with the code, so a front-end gap is loud.
- What: keep call-time checks in the oracle for values computed from attributes (`V` rows).
- Benefits: about 12 checks fewer in the backend and 8 in the oracle; one rule, one place.
- Caveats: the oracle's runtime copies of shape rules stay for expression shapes (O7).
- Fail points: a front-end rule weaker than the generator's; noticed by running the old and new verdicts side by side on every test spec before deleting.
- Order and dependencies: D4, B14, B23, O6, O7.
- Verification: refusal tests of both generators green with code-based assertions; `grep -n "raise ValueError" act_backend/spec_checks.py` lists only `U` rows.
