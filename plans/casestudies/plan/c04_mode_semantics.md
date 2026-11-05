# C4 Mode-dependent semantics

- What: A semantics-level case on an attribute or state value, each arm a semantics block, so a generator sees one arm per mode.
- Benefits: Replaces `select` over every mode in 7 case studies; the backend can select an instruction per mode instead of always computing all (gap rank 5).
- Caveats: Arms must agree on output shapes; this is the first control construct in a semantics and touches S7 and F.
- Fail points: An arm left unreachable by constraints hides a wrong semantics; noticed by oracle coverage counting arms hit by random tests.
- Order and dependencies: L and F, then B and O, then T.
- Verification: Gemmini `mvout_acc` and Tensix `mvmul` rewritten with a case give the same oracle results as the `select` versions.
