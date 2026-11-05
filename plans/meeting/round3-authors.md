# Round 3: authors apply the decisions

Each track author revises its own folders so the plans state the decisions in force.

## Binding inputs

- Owner decisions: plans/minutes/owner-decisions.md, 1 to 33.
- Panel decisions: plans/meeting/panel-1/decision.md to panel-8/decision.md, where they rule.
- plans/minutes/rulings.md holds draft answers only; cite panel decisions (for example Q4.6), never an R id, as the reason for a change.

## Items the panels sent to the owner, now settled

- Oracle shape rules are copied into each generated oracle at generation (owner 25).
- The oracle does not pivot to fsim-lib in this plan; the fsim analysis is a follow-up direction (owner 23).
- Versions: taidl, taidl-to, act 3.0; act-backend 2.0 (owner 26).
- Backend search: wall-clock bound stays the user default, deterministic mode for tests, never a wrong answer (owner 27).
- New dependencies allowed when their item lands: cbindgen, hypothesis, mutmut; no StableHLO interpreter (owner 28).
- rustfmt runs on generated crates (owner 29).
- The user's file is the "ISA specification"; the language document is the "TAIDL language specification" (owner 32).
- Releases follow the act 2.x structure: released pins only, no pre-release tags, no commit or branch pins in public (owner 33).
- A backend never assumes an unsupported input away: it rejects it with a coded error (owner 27, never a wrong answer).
- Where a clause id sits in a message: line 1, as panel 3 recommends, until the owner says otherwise.

## Pending panel 9

- Everything about overlap between inputs and outputs, in-place marks, placement marks, and read-before-write ordering waits for panel 9.
- Owner decision 31 already binds: TAIDL assumes no hazard model; the ISA author declares overlap behavior.
- Do not edit those passages beyond marking them "pending panel 9"; remove any text that states read-before-write as a language-wide rule.

## Identifiers

- Shared-code items S1 to S9 become U1 to U9 everywhere; spec section ids S1 to S10 and S1.1 and so on stay.
- Decide by context which "S<n>" means a shared item; dependency columns hold item IDs only.

## Writing rules

- Replace text in place; docs state what is, with no history or meeting narration.
- ASCII only; one sentence per line; bullets by default; tables for per-item data.
- Files under about 150 lines; one README.md index per folder; only README.md files link.
- No git commands; read-only shell; write only in the folders named in your brief.
