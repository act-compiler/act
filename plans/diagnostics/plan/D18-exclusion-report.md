# D18 Exclusion report

- What: an exclusion is a diagnostic at severity `warning` with a `U` code, the section id, the instruction, and the author's file and line (scheme.md).
- What: the backend prints one line per exclusion at generation and stores the same lines in the generated crate (B46).
- What: an ISA specification with no instruction left rejects with `UB019`; an unknown key in the backend's `meta` namespace rejects with `UB020`.
- What: the reader of the generated backend reads the stored report and stops with `KB042`, naming the instruction and its reason, when a kernel needs an excluded instruction.
- What: `registry-seed.md` marks every `U` row as exclusion or rejection; the generated unsupported table (D11) shows the outcome per row.
- What: the oracle excludes nothing and has no row.
- Benefits: an instruction is never dropped silently; the architect sees what the target left out and the line to change.
- Caveats: an exclusion is not a stop; each example lists its expected exclusions, and an unlisted one fails the test.
- Fail points: a reason printed and not stored; noticed by a test that reads the crate.
- Order and dependencies: D1, D2, D6, B46.
- Verification: a test per exclusion row asserts code, instruction, and author line in the printed and the stored report; a kernel that needs the excluded instruction ends with `KB042`.
- Breaking: yes for the backend's exit behavior on a specification that used to be rejected.
