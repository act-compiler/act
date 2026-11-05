# Debate process

Every engineering question is settled by a panel of three debaters, one each of Sonnet, Opus, and Fable, and a separate moderator.
The chair sets the agenda and runs the rounds; the chair decides nothing.
Only the owner decisions bind a panel; a draft answer is one option among others.
An owner decision is settled and never debated; a debater with a strong, evidence-backed counter writes it under "Escalation to the owner", and the moderator forwards it unanswered.

## Inputs

- Owner decisions: plans/minutes/owner-decisions.md (binding).
- Draft answers: plans/minutes/rulings.md (an option to weigh, never a default).
- The panel's agenda: plans/meeting/panel-N/agenda.md.
- Track plans, reviews, and review responses: every folder under plans/.

## Evidence

A claim counts only with a citation; the moderator gives an uncited claim no weight.

| Kind | Where |
| ---- | ----- |
| Code | /home/user/taidl, /home/user/act-backend, /home/user/act-oracle, /home/user/act, cited as file:line |
| Owner papers | ACT, OOPSLA 2026: https://act-compiler.github.io/assets/pdf/2026-oopsla-act.pdf |
| Owner papers | TAIDL, MICRO 2025: https://act-compiler.github.io/assets/pdf/2025-micro-taidl.pdf |
| StableHLO | https://openxla.org/stablehlo/spec and github.com/openxla/stablehlo |
| XLA | https://openxla.org/xla/operation_semantics and github.com/openxla/xla |
| MLIR | https://mlir.llvm.org/docs/ (linalg, tensor, memref, vector, arith, ArmSME, AMX, nvgpu) |
| PyTorch | https://pytorch.org/docs/stable/ (dtypes, ATen, torch.fx) |
| AWS Neuron NKI | https://awsdocs-neuron.readthedocs-hosted.com (nki.isa, nki.language) |
| Compiler practice | egg, egglog, Cranelift ISLE, LLVM GlobalISel and TableGen documentation |
| Meeting evidence | plans/casestudies/, plans/priorart/, plans/semantics/, plans/reviews/ |

- Web pages are read with WebSearch and WebFetch, loaded through ToolSearch.
- A debater researches before taking a position, and cites at least one external source on any question about vocabulary, semantics, or compiler practice.

## Rounds

1. Positions: each debater answers every agenda question alone, without reading the others.
   Per question: position, argument as numbered steps, evidence, confidence, and what evidence would change the position.
2. Rebuttals: each debater reads the other two round-1 files.
   Per question: the strongest opposing argument restated fairly, then a rebuttal or a concession with evidence, then the final position.
3. Decision: the moderator reads all six files and rules per question on the quality of evidence and reasoning, never on the count of votes.
   Per question: decision, the reasoning chain, the evidence it rests on, dissent, and confidence.
   A question whose evidence is balanced and whose answer is a value judgment goes to the owner, with the options.

## Files

| Round | File |
| ----- | ---- |
| Agenda | plans/meeting/panel-N/agenda.md |
| Positions | plans/meeting/panel-N/round1-sonnet.md, round1-opus.md, round1-fable.md |
| Rebuttals | plans/meeting/panel-N/round2-sonnet.md, round2-opus.md, round2-fable.md |
| Decision | plans/meeting/panel-N/decision.md |

## Writing rules

- ASCII only; one sentence per line; bullets by default; tables for per-item data.
- State what is, with no narration of the meeting itself.
- Only README.md files link to other files; questions are named by ID, such as Q3.4 (panel 3, question 4).
- A debater writes only its own file; no git commands; only read-only shell commands.
