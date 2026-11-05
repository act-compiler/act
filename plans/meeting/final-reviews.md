# Final reviews

Two review tracks after the panels: a management review of the whole plan, and a gap hunt on the TAIDL language specification.
Reviewers judge; they do not pivot the plan, and they propose changes only with evidence.
Owner decisions bind (plans/minutes/owner-decisions.md); panel 12 is still deciding, so read its research and round files for the long-term platform.

## Management reviewers

- Read plans/README.md, the roadmap (waves, milestones, risks), the owner decisions, the panel decisions, reviews/approval.md, and each track's plan index.
- Judge against the product goals: robustness (nothing fails without a genuine cause; no undefined behavior), reusability (one front end, one contract, shared corpus), retargetability (any accelerator in scope, new generators easy to add), usability for architects, and maintainability.
- For the timeline: critical path, blockers, priorities, value and impact per milestone, and what is under-specified or mis-ordered.
- Write plans/reviews/management/<lens>.md, under 150 lines: verdict, findings table `ID | Area | Finding | Evidence | Recommendation | Severity`, and the top five actions.

## Spec gap hunters

- Read the whole TAIDL language specification draft in plans/language/spec-draft/ (start at README.md), then your slice in depth, with the front end at /home/user/taidl and the panel decisions as reference.
- Find: statements two careful readers could implement differently; cases with no stated outcome; contradictions between pages or with panel or owner decisions; missing pieces an architect or a generator author needs; examples that do not follow the rules.
- For each finding, propose exact replacement or added text, and say whether it needs a debate (a real design choice) or is a clarification.
- Write plans/reviews/spec-gaps/<slice>.md, under 150 lines: table `ID | Section | Gap | Proposed text | Needs debate?`.

## For every reviewer

- Cite file:line or a URL for each claim; ASCII only; one sentence per line; no git; read-only shell; take up to 25 minutes.
- Final report: one line, at most 30 words.
