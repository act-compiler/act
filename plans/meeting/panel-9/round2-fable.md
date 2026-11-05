# Panel 9, round 2, Fable: overlap and hazard declarations

Lens: how shipped ISAs behave when an instruction's operands alias.
Every position is tested against owner decisions 16, 17, 22, 24, 30, 31, and 34.

## The decision-31 test applied to all three round-1 files

- All three files compute the value of a legal overlap from the inputs as they were before the issue.
- Decision 31 forbids the language assuming a hazard model such as read-before-write; it does not forbid the language having an evaluation model.
- The pre-issue read is the evaluation model of a dataflow semantics (Q1.1 C; instruction_codegen.py:162-181): it says when the model samples storage, not what the hardware does.
- A hazard model is a claim about the hardware's value when the same storage is read and written, and decision 31 leaves that claim to the author's declaration.
- The author declares two things: which overlaps are legal (constraints) and what value the instruction produces on them (the semantics body, a function of the pre-issue inputs and the attributes).
- So a legal overlap's value is "the author's semantics on the pre-issue state", not "read-before-write"; the two coincide only when the semantics is written as if order did not matter.
- Sonnet's Q9.3 crosses the line: it makes every undeclared overlap legal with the read-before-write value, which installs the one hazard model decision 31 names as the language-wide default.
- Opus's Q9.1 step 3 ("a dataflow semantics can define exactly one value, the snapshot") and my own Q9.1 step 4 overstate: the snapshot is the value of order-independent semantics, and an author can write an order-dependent result as a pure function of pre-state and offset.

## Real hardware where overlap is not read-before-write

| Case | Documented behavior | Source |
| ---- | ------------------- | ------ |
| x86 `REP MOVSB`, destination one byte above source | The result equals that many separate byte moves, so the destination fills with the first byte; fast-string mode is entered only when source and destination do not overlap within a cache line (SDM Vol. 3 section 8.2.4 as mirrored) | Intel SDM fast-string conditions and optimization manual, via search results |
| Arm FEAT_MOPS `CPYF*` versus `CPY*` | `CPY` tolerates overlap and picks the copy direction, memmove-style; `CPYF` always copies forwards, memcpy-style, so an overlapping `CPYF` gives the ordered result, not the snapshot | QEMU commit "target/arm: Implement the CPY* instructions"; Linux arm64 MOPS memcpy patch |
| RVV `vslideup`, `vrgather`, `vcompress` | Destination overlap with the source is reserved, with the spec's note that this "avoids WAR hazards on the input vectors"; the hardware's natural value is not the snapshot, so the ISA forbids the case | RVV spec 1.0, slide and gather sections (fetched round 1); Sonnet E2 |
| RVV narrowing and widening | Overlap is legal only at the fixed low or high part, exactly the positions where in-order element processing reproduces the snapshot | RVV spec 1.0, Vector Operands (fetched round 1) |
| C `memcpy`, CUDA `cudaMemcpy` | Overlap is undefined | C11 7.24.2.1; CUDA runtime documentation (both from memory) |

- Pattern: register-file ISAs (RVV equal EEW, AMX `tsrcdest`) and memmove-style copies give the snapshot; forward bulk copies give the ordered value; permuting units reserve overlap.
- A shipped ISA legalizes overlap exactly where its element order reproduces the snapshot, and otherwise reserves it or documents the ordered value.
- What an ISA author does in TAIDL for each class: snapshot hardware writes plain semantics; `REP MOVSB` or `CPYF` hardware writes the ordered result as a function of the pre-state and the offset (for a forward copy at offset `k`, element `i` is source element `i mod k` over the overlapped stretch), or declares the pair apart; reserved or undefined hardware declares the pair apart, which turns the reserved encoding into a defined fault (decision 16).
- No tensor accelerator case study documents a defined non-snapshot overlap; the two documented cases are CPU bulk copies, so the ordered-copy class has no v3 site and stays expressible, not sugared.

### Q9.1

Strongest opposing argument (Opus): three relations, apart, may overlap, and tied, with apart as the default because it alone assumes no hazard model, and every finer rule is a constraint.
Also (Sonnet): the declarable set cannot be limited to input-output pairs, since AMX raises #UD when any two of the three tile operands are equal.

- Concession to Sonnet: input-input distinctness is a real ISA rule (AMX E3) and is a hand-written constraint such as `@a.s1 != @a.s2`, so the closed set is the set of marker relations, not the set of all declarable facts.
- Concession to Opus: "no partial overlap" and the RVV positional rules are "may overlap" narrowed by a constraint; the marker stays as shorthand under decision 30, not as a distinct relation.
- Rebuttal to Opus on "may overlap carries no information": with an implicit default constraint on every unmarked pair (Q9.3), the may-overlap marker is the only way to lift it, so it is a declaration with content, which Opus's own Q9.2 step 5 admits.
- Rebuttal to all three files, mine included: the value under a legal overlap is the semantics on the pre-issue state, and the table above shows the author must sometimes write the ordered result; the relations declare legality, the body declares value.
- Option B (a hazard-order declaration) stays out of v3: it would need a sequential evaluation model against Q1.1 C, and its only sites (`REP MOVSB`, `CPYF`) are outside decision 15's scope; the closed form or `set_apart` covers them today.
- Changed: the fourth relation is kept, but its meaning is "no implicit constraint on this pair", with the value left to the semantics body.

Final position: D, three marker relations (tied, apart, exact-or-apart) plus a may-overlap marker that lifts the implicit default, input-input rules as hand constraints, and no hazard-order declaration in v3.
Confidence: medium-high.
Still divides the panel: whether may-overlap is a marker (Fable, Opus medium) or unnecessary because undeclared overlap is already legal (Sonnet).

### Q9.2

Strongest opposing argument (Opus): name two predicates, `apart(R, S)` and `same(R, S)`, rather than expanding to per-dimension inequalities, because `same` must compare sizes too (F5) and the backend needs the pair relation without pattern-matching text.

- Concession on F5: `keep_off` compares starts only (instruction.py:212-222), so equal starts with unequal sizes pass `set_no_partial_overlap` although the regions overlap partially; the exact relation is equal starts and equal sizes.
- Concession on replacement: a marker that permits overlap cannot append beside an implicit constraint (instructions.md:38), so a pair named by an overlap marker gets the marker's expansion in place of the implicit one; `keep_off` already replaces its own earlier text (instruction.py:227-231), so the mechanism exists.
- Rebuttal on named predicates: the expansion text is what owner 30 promises the user can read and replace, and the allocator consumes a disjunction over offsets (constraint.cc:88-103), so the per-dimension text stays the canonical form; a predicate name is a front-end convenience that must expand to the same text.
- Rebuttal to Sonnet's step 10 and Opus's step 7: the identical-text shortcut skips a dimension only when both texts are identical, and a differing text that evaluates equal under decision 34 is checked at issue, so both files and mine agree.
- Unchanged: the tie rides on `set_inplace` as a datum beside its `same` constraint, because the copy pass and the allocator read the pairing (inplace_fix.rs:59-80; constraint.cc:214-230), and no tool recovers it from `==` text.

Final position: D under owner 30, every marker expands to constraint text over evaluated start and size terms, an overlap marker replaces the implicit constraint for the pair it names, `same` includes sizes, and the tie is the one recorded datum.
Confidence: medium-high.
Still divides the panel: named predicates versus expanded text as the canonical form; the oracle and backend behavior is the same either way.

### Q9.3

Strongest opposing argument (Sonnet): constraints only remove scenarios, so the default must be the largest legal set or an author cannot say the hardware allows an overlap, and `vnsrl.wi v0, v0, 3` is a legal partial overlap a default fault would reject with no opt-in.
Strongest opposing argument (Opus): the default is an implicit `apart`, the only default that assumes no hazard model, and an undeclared overlap faults like any constraint.

- Rebuttal to Sonnet's subtraction premise: a marker may replace an implicit constraint for the pair it names (Q9.2), so the default is not forced to be the largest set, and `set_inplace` already must displace any implicit apartness for its pair.
- Rebuttal to Sonnet on RVV narrowing: the RVV spec states low-part narrowing as a specific rule, not as a default, and in TAIDL it is the declared constraint `out_start == in_start` with a smaller output; the ISA itself did not make it the default.
- Rebuttal to Sonnet on decision 31: "any undeclared overlap reads before it writes" is the read-before-write hazard model installed language-wide, which is the sentence decision 31 rules out.
- Rebuttal to Sonnet's step 6 and Q9.4 step 6 together: a systolic ISA whose author omits `set_apart` would pass the oracle and the backend would place a dying input partly over its output (constraint.cc:206-236), so a silently wrong kernel ships; decision 27 says a wrong answer is never acceptable.
- Rebuttal to Opus's `apart` default: decision 22 says `r0 = r1 + r2` "allows r2 to equal or differ from r0" without a mark, so an unmarked exact overlap is a legal scenario by the owner's own word, and `apart` would fault it unless marked.
- Rebuttal to Opus's neutrality claim: `apart` as default assumes the hazard model of a reserved encoding, which is as much a hardware assumption as the snapshot; no default is neutral, so the choice falls to decisions 16, 22, and the shipped evidence.
- Evidence for exact-or-apart: every shipped ISA that legalizes exact overlap gives the snapshot (RVV equal EEW, AMX `tsrcdest`), and even the ordered copies give it on exact overlap, since a forward copy onto itself is the identity, so no hazard model is assumed for the exact case.
- Evidence for partial as fault: no tensor ISA permits partial overlap by default (RVV reserves; `CPYF`, `memcpy`, and `cudaMemcpy` leave it undefined), so a default fault fails no scenario a real architecture allows by default, and the may-overlap marker admits the ones an author's hardware does allow.
- Changed: with the F5 concession, "exact" means equal starts and equal sizes, so equal starts with a smaller output is partial and faults unless declared, which is how RVV narrowing is handled.
- Unchanged: a violated constraint of any origin is one fault at issue (S1.3.5), static rejection only where text alone settles it (cross-buffer tie, decision 24; unequal size text on a tie).

Final position: C, the stated default is an implicit exact-or-apart constraint per unmarked same-buffer input-output pair; exact overlap is legal with the semantics' value, partial overlap faults, the may-overlap marker lifts the constraint, and every violation is the one constraint fault.
Confidence: medium-high.
Still divides the panel: three defaults; if decision 22 is read as a statement about the backend's placement freedom rather than the model's legal set, Opus's `apart` becomes permissible and the choice between apart and exact-or-apart is a value judgment for the owner.

### Q9.4

Strongest opposing argument (Opus): the allocator's touch exception (constraint.cc:206-210) silently assumes read-before-write for every unmarked pair whose input dies at the instruction, so the implicit constraint must reach the allocator, or the exception must apply to marker-named pairs only.

- Concession: the touch exception is the backend's own hazard assumption, forbidden by decision 31 for unmarked pairs; constraint.cc:207 lets `[a,b]` and `[b,c]` share storage at any offset, and only the tied check at 214-230 narrows it.
- Under the exact-or-apart default the fix is the one I gave in round 1: the implicit constraint enters `overlap_constraints` as a disjunction (constraint.cc:88-103), so a dying input lands exactly on or apart from the output, never partly over it.
- Rebuttal to Sonnet's step 6: the backend may not leave correctness to the author's omission; it must honor the implicit constraint as it honors bounds.
- Oracle: unchanged from round 1, every constraint evaluated before any load, value from the semantics on the pre-issue state (instruction_codegen.py:162-181); an author's ordered-copy formula runs through the same path with no new code.
- Corpus, adopting Opus's matrix: five declarations by three placements, each expecting a value or the one fault, plus the F5 case (equal starts, unequal sizes, fault by default), the decision-34 case (starts differing in text, equal in value, negative attribute), the RVV widening case as a hand constraint, and the AMX input-input case as a hand constraint.
- Corpus, backend: per declaration, an input dying at the instruction, asserting the allocator overlaps only where the constraints permit and the oracle agrees on the value.
- Corpus, expansion check: marker and hand-written text give the same fault set over a swept attribute range (decision 30's promise).

Final position: both generators read one constraint list with the implicit exact-or-apart entries; the backend narrows its touch exception to that list plus the tie; the corpus runs the declaration-by-placement matrix with the F5, decision-34, RVV, and AMX rows.
Confidence: medium-high on oracle and corpus; medium on the backend, where the implicit disjunction in the solver is unlanded.
Still divides the panel: nothing material once the Q9.3 default is settled; the matrix adapts to whichever default the moderator rules.

## Escalation to the owner

- None against a decision.
- Decision 22's sentence "r0 = r1 + r2 allows r2 to equal or differ from r0" is read here as making an unmarked exact overlap a legal scenario of the model; if the owner meant only the backend's freedom to reuse storage, the default for an unmarked pair is between apart (Opus) and exact-or-apart (Fable), and the owner should pick, since the evidence is balanced and the choice is a norm preference.
- Decision 31 is met by all three files once the value of a legal overlap is read as "the author's semantics on the pre-issue state"; the owner may want that sentence in the norm, so that an author whose hardware copies forward knows to write the ordered result or declare the pair apart.
