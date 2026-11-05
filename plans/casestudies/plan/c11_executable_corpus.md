# C11 Case studies as an executable corpus

- What: Turn each sketch into a runnable ISA description with oracle-generated tests, tagged by the workaround each instruction uses.
- Benefits: A regression suite of 10 real-ISA shapes for front end, oracle, and backend; each language item shows its effect as tags disappear.
- Caveats: The sketches are not vendor-verified; expected values come from the oracle, not hardware.
- Fail points: The front end rejects a sketch for a reason the case study missed; noticed on the first run, which is the point.
- Order and dependencies: T and O.
- Verification: All sketches pass the front end and generate an oracle; the count of workaround tags is tracked per release.
