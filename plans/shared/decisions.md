# Decisions: where shared semantics live

The question: how much of what both generators derive moves into the front end, and how the repos release together.

## Positions

| Role | Position | Strongest argument | Weakest point |
| --- | --- | --- | --- |
| Consolidator | Move shapes, element types, integer folding, literal checks, and the final region checks into `taidl`; generators keep only emitters and their own unsupported-feature refusals. | X1, X3, X4, X5, X6 are each one rule implemented once, twice, or not at all; a rule in the front end makes every generator accept the same set (D2, D26 of the spec track). | A larger front end becomes a release bottleneck for two generator teams. |
| Decoupler | Keep generators independent; share only a serialized intermediate form and a conformance corpus of ISA specifications with expected verdicts. | A serialized form ends the reliance on ANTLR internals and dict order (X7), and lets a non-Python generator exist. | A corpus detects divergence after the fact; it does not stop a third copy of shape rules being written. |
| Judge | Consolidate semantics, decouple representation: one checker in the front end, one typed model exposed by it, plus the corpus as the test. | Owner goals are simplicity and generators that follow the spec; a shared checker is the simplest way to the second, a corpus the cheapest proof. | A serialized format now is speculative: both consumers are Python today. |

## Verdict

| ID | Question | Ruling | Rationale |
| --- | --- | --- | --- |
| SD1 | Who checks shapes and element types per word? | The front end (U4), seeded from `act-oracle/taidl_to/generic/shapes.py` and `act-oracle/taidl_to/oracle_visitor.py:19-97`; the generator copies the module into each generated oracle, stamped with its taidl version. | Only one complete copy exists; moving it closes X1 and X5 for literal and foldable dims. The copy keeps the oracle free of a run-time `taidl` import. |
| SD2 | Who checks region values against their data model? | The front end, in one final validation call (U3) that every generator makes first (U10). | S7.2 is a language rule (taidl/docs/semantics.md:47); X3 and X4 show both copies wrong in different ways. |
| SD3 | Who folds integers and checks literals per type? | The front end exposes `fold` and `literal_fits` as listed contract functions; generators call them. | Three folders and two literal checks disagree today (X6, X8). |
| SD4 | Who writes code for expressions? | Each generator. | Python, Rust, C++ and solver emitters differ by target; only the tree walk is common. |
| SD5 | Serialized intermediate form? | Not now; a typed Python model and a small typed node model, with `CONTRACT_VERSION` and one minor release of deprecation per member; ANTLR trees stay reachable and unstable. | Both generators are Python; add a serialized form at the second non-Python consumer. |
| SD6 | Conformance corpus? | Yes, in `taidl`, each ISA specification with its expected verdict per generator. | It is the decoupler's best idea and the proof of SD1 to SD3. |
| SD7 | Unsupported-feature refusals? | Stay per generator, keyed by spec section (L7); a backend never assumes an unsupported input away and rejects it with a coded error. | They describe a generator, not the language, and a wrong answer is never acceptable. |
| SD8 | Release coordination? | One v3 release train: front end first, both generators pin its released version, the umbrella's version check gates the tag; taidl, taidl-to, and act are 3.0, act-backend is 2.0. | Matches the owner's one coordinated major release and the act 2.x structure of released pins. |
| SD9 | Docker and CI scripts? | One copy in `act`, consumed by tag; repos keep thin wrappers. | Four near-identical copies differ only by prefix today. |
