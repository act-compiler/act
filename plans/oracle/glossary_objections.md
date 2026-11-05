# Objections to the glossary

From the act-oracle track, against the language glossary (L25).

| Term | Objection | Suggested resolution |
| --- | --- | --- |
| off-chip memory (bans HBM) | Owner decision 19 keeps `hbm`; the kernel keyword is `hbm` (`taidl_to/generic/decorator.py:142`). | The glossary allows `hbm`. |
| instruction attribute (bans parameter) | The front end itself stores addressing attributes as `Instruction.parameters` and computational ones as `Instruction.comp_attr` (read at `taidl_to/instruction_codegen.py:149-157`). | Rename in the front end first (L track), then generators follow; until then allow the field names. |
| data model (bans buffer) | Owner decision 19: a data model has buffers, so "buffer" names its storage (`taidl_to/generic/utils.py:698-713`). | The glossary allows "buffer" for a data model's storage. |
| region (bans tile, window) | Withdrawn: the terminology reference gives "region value" for the oracle's tile and "region" for its window, which covers both uses. | Add "region value" to the glossary, since the reference uses it and the glossary does not define it. |
| reject (bans refuse) | Accepted, but "rejected" must also cover errors raised at a call, which the oracle raises at kernel compile time, not at generation. | Add "reject at the call" as a defined use, or a term for call-time checks. |
| generator vs oracle | The glossary has no term for the generated artifact, the oracle (`api.py`, `decorator.py`) that a kernel runs on. | Add "oracle: the artifact TAIDL-TO generates" so docs need not say "generated oracle" and "simulator" interchangeably. |
| implementation-defined | Owner decision 16 leaves no undefined behavior; each S1.3 case is a defined error or a stated norm. | Drop the term, or keep it only for a generator's own API outside the language. |
