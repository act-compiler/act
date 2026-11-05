# Objections to the glossary

From the act-oracle track, against the language glossary (L25).

| Term | Objection | Suggested resolution |
| --- | --- | --- |
| off-chip memory (bans HBM) | `kernel(hbm, input, output, constant)` takes `hbm` as a public keyword (`taidl_to/generic/decorator.py:142`); renaming it breaks every kernel. | Allow `hbm` as the parameter name of a generator API, as `var_type` is allowed for `add_data_model`. |
| instruction attribute (bans parameter) | The front end itself stores addressing attributes as `Instruction.parameters` and computational ones as `Instruction.comp_attr` (read at `taidl_to/instruction_codegen.py:149-157`). | Rename in the front end first (L track), then generators follow; until then allow the field names. |
| data model (bans buffer) | The oracle's errors say "buffer" for a data model (`taidl_to/generic/utils.py:698-713`, `README.md:122`). Accepted: the oracle should switch. | No objection; plan item O15. |
| region (bans tile, window) | Withdrawn: the terminology reference gives "region value" for the oracle's tile and "region" for its window, which covers both uses. | Add "region value" to the glossary, since the reference uses it and the glossary does not define it. |
| reject (bans refuse) | Accepted, but "rejected" must also cover errors raised at a call, which the oracle raises at kernel compile time, not at generation. | Add "reject at the call" as a defined use, or a term for call-time checks. |
| generator vs oracle | The glossary has no term for the generated artifact, the oracle (`api.py`, `decorator.py`) that a kernel runs on. | Add "oracle: the artifact TAIDL-TO generates" so docs need not say "generated oracle" and "simulator" interchangeably. |
| implementation-defined | 64-bit overflow is implementation-defined in S1.3 per S6.2; the glossary should name where each generator documents its choice. | Point to each generator's `docs/coverage.md`. |
