# Risks

The risk register for the waves and milestones (Q8.11).
Each risk has an observable tripwire and names the exit criterion it endangers.
A response moves a date or reorders work; it never cuts scope.
A risk that no exit criterion covers adds a criterion.

## Categories

- schedule: the work takes longer than the waves assume.
- technical: a design choice may not hold.
- dependency: an input from another track, panel, or the owner arrives late or changes.

## Register

| ID | Category | Risk | Tripwire | Exit criterion endangered | Response | Owner role |
| -- | -------- | ---- | -------- | ------------------------- | -------- | ---------- |
| K1 | schedule | The assist factor is lower than the 4.3 the weights need | The wave 1 ratio of unassisted to assisted weeks falls below 4.3 | M1 dates set from wave 1 measurements | Move dates at the M1 rebaseline; keep scope | roadmap author |
| K2 | schedule | Graph integrity drifts as plans change | The Q8.1 checker fails, or the waves.md check line no longer holds | Standing graph-checker criterion | Regenerate the tables, then reassign waves before more work starts | roadmap author |
| K3 | technical | B7's Rust shape rules diverge from the taidl shape module | A shape rule exists in two places without the agreement test | M2 agreement test | Write the agreement test before U4 closes; the taidl module wins | backend lead |
| K4 | schedule | B21 and B22 land checks that D5 then deletes | D5 deletes a check landed in the same quarter | M3 duplicates deleted | Land B21 as a call to `check` only, with no new check text | backend lead |
| K5 | technical | v3 migration leaves users without a path | A rejected v3 construct has no replacement and no clause id (RA11) | M5 cold-start trial | Add the replacement and clause id to the L30 rename table and D15 | language lead |
| K6 | technical | 64-bit checking is not enough (R56) | Generated Rust needs `i128` or bignums | M3 front-end checks; F3 hold | Hold F3 until R56 is ruled; widen only the failing operators | semantics lead |
| K7 | technical | Generators depend on the ANTLR parse tree (SD4) | A generator imports a parse-tree type | M2 generators read only the `Model`; M4 string surface deleted | Route the import through the checked `Model` (U10) before U14 | shared-code lead |
| K8 | schedule | Waves 3 and 4 carry the most weight | Recomputed wave weights put wave 3 or 4 above twice the wave 1 measured throughput | M2 and M3 | Move dates; split the wave by milestone exit, never drop items | roadmap author |
| K9 | dependency | A later fsim-lib migration reorders oracle work (owner 23) | The O29 plan changes an oracle edge that feeds an exit criterion | M2 unroll, M3 oracle checks | Keep the current path to M4; schedule the migration after M5 | oracle lead |
| K10 | dependency | The release order is written too late | No A16 checklist when taidl 3.0 is cut | M4 release order | A16 is a wave 1 gate; U8 waits for it | act lead |
| K11 | dependency | Later items enter waves without a decision | The checker finds an other-core.md item in a wave without a stated reason | Standing graph-checker criterion | State the reason in its wave row, or move it to wave 7 | roadmap author |
| K13 | technical | Backend pair relations do not hold under the narrowed touch exception (Q9.4, medium confidence) | A T31 row allocates a pair outside its relation, or B11's copy changes a chosen kernel's goldens | M2 allocation per relation, M4 backend overlap rows | Fix B30 and B40 before B11 and B19; the oracle's overlap matrix stays the reference | backend lead |
| K14 | dependency | Held items wait on rulings | R20 or R56 is open when its wave starts | M3 front-end checks for F3; wave 7 for P6 | Run the wave's other items first; slot the held item when ruled | roadmap author |
| K15 | dependency | The Ascend study has no table ID | No case-studies item for R15 exists when wave 2 starts | Wave 1 gate row; R59 counts in wave 3 | The case-studies track adds the item and the tables regenerate | casestudies lead |
| K16 | dependency | The unroll spike misses the Q5.1 bound | O19 measures over 60 s on the stress kernel or over 2x on a loop test | M2 unroll criterion | O11 lands as the named fallback with its equivalence test | oracle lead |
| K17 | technical | A public pin points at an unreleased version | A public manifest holds a pre-release tag, commit pin, or branch pin | M4 negative criterion | Release the dependency first, in the A16 order, then pin it | act lead |
| K18 | schedule | Outside people for the cold-start trial are not found in time | Two trial participants are not named when wave 6 starts | M5 cold-start trial | Move the M5 date; the trial is not replaced by a team-internal run | act lead |
| K19 | dependency | The front-end contract churns after generators adopt it | U2's contract version changes after M2 | M3 and M4 | Batch contract changes into taidl 3.0; generators adopt once | shared-code lead |
| K20 | schedule | The encoding unit is the last v3 work and carries two L items (B45, T33) | The unit is incomplete when every other wave 5 item is done | M4 encoding unit criterion | Move the v3 date; the unit never ships in part and is never deferred to a 3.x release (Q10.4) | language lead |
| K21 | technical | The allocator cannot hold a declared attribute domain or a derived link | B38 or B44 generation reports a domain the solver cannot hold | M1 width layer, M4 encoding unit | A loud generation error, never a silent clamp (Q10.1); fix the solver domain before B45 | backend lead |
| K22 | technical | A generator still stops without a genuine cause (owner 40) | A T36 case stops on a valid ISA specification, assembly input, or kernel | M3 failure policy, M4 failure policy cases | Fix the raise site under D16, O31, or B41; dates move | diagnostics lead |
| K24 | technical | The traced `Model` differs from the string-built `Model` on a corpus file (Q11.4) | A T37 file fails equality up to call sites | M4 equivalence and string-surface deletion | Fix the tracer or the converter; L45 and U14 wait until every file is equal | language lead |
| K25 | technical | An exclusion hides an instruction the architect expected to work (Q11.3, medium-high confidence) | A kernel is rejected at compile time naming an excluded instruction that the expectations table marks supported | M3 exclusion report and failure policy | Fix the B14 table or the expectations row; never turn the exclusion into a silent drop | backend lead |
| K26 | schedule | The tracer is the one L item on the M2 path of the front end (L42, U12) | L42 is open when every other wave 3 front-end item is done | M2 traced surface criterion | Move the M2 date; the converter and the deletion wait, never the order of Q11.4 | language lead |
