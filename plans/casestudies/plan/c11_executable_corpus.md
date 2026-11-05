# C11 Case studies as an executable corpus

- What: Turn each sketch into a runnable ISA specification with oracle-generated tests, tagged by the workaround each instruction uses.
- Boundary: P7 holds one example per operation row and per rejection row; C11 holds accelerators.
- Benefits: A regression suite of real-ISA shapes for front end, oracle, and backend; only a sketch that ran counts as a site under Q2.3.
- Checks: the Hopper fragment permutation against the PTX fragment figure; the five `wgmma` immediates; the shape bugs fixed by `reshape`.
- Caveats: The sketches are not vendor-verified; expected values come from the oracle, not hardware.
- Fail points: The front end rejects a sketch for a reason the case study missed; noticed on the first run.
- Order and dependencies: T and O.
- Verification: All sketches pass the front end and generate an oracle; the count of workaround tags is tracked per release.
