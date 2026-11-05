# Objections to the glossary from act-backend

Each objection names the glossary term, the clash, and a suggested resolution.

| term | objection | suggested resolution |
| --- | --- | --- |
| start (bans offset, address) | The backend places each kernel argument at a byte offset in `hbm`, a concept the glossary does not cover and which is not a region start. | Scope the glossary to spec concepts; let a generator define "hbm offset" for its own placement. |
| constraint (bans precondition) | An e-graph rewrite has a condition under which it fires; egg calls it a condition, and it is not a TAIDL constraint. | Allow "rewrite condition" in generator internals; ban "precondition" for TAIDL constraints only. |
| value (bans tensor, variable) | The backend reads an XLA-HLO kernel whose values are tensors and whose rewrite patterns have variables; neither is a TAIDL value. | State that banned synonyms apply when naming TAIDL concepts, not the kernel IR or e-graph terms. |
| buffer | Owner decision 19 makes "buffer" the term for each declared memory of a data model; the Rust enum `Buffer` already uses it. | List "buffer" in the glossary; keep `Buffer`; drop the ban on "buffer". |
| reject (bans refuse) | No objection: act-backend adopts reject. | Rename planned (B27). |
| unsupported feature (bans limitation, gap) | No objection; `docs/coverage.md` uses "rejected" and "partial" as statuses, which the glossary may want to list. | Add `supported`, `rejected`, `partial` as the status words of S1.2. |
