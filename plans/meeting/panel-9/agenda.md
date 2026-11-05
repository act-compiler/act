# Panel 9 agenda: overlap and hazard declarations

Joint follow-up to panels 1 and 2, which split on the in-place mark.
Binding owner decisions: 16 (no undefined behavior), 17 and 22 (in-place reuses the marked input's storage), 24 (cross-buffer in-place is rejected), and 31.
Owner decision 31: TAIDL assumes no hazard model such as read-before-write; the ISA author declares how an instruction behaves when its inputs and outputs overlap, and the language must not fail scenarios real architectures allow.
The owner also asked: if one marker can be expressed as constraints, does the same hold for the other markers?

### Q9.1: What must an ISA specification be able to declare about overlap between an instruction's inputs and outputs?

- Context: candidates include "output must reuse input k's storage", "output may reuse it", "these regions never overlap", "no partial overlap", and an ordering guarantee (inputs read before outputs written) or its absence.
- Options: A. the three current markers' meanings only; B. A plus an explicit hazard-order declaration; C. a small closed set of overlap relations chosen from real ISAs; D. other, argued from evidence.
- Evidence: panel-1/decision.md Q1.3, panel-2/decision.md Q2.1 and Q2.2, their round files; taidl/taidl/instruction.py (set_inplace, set_apart, set_no_partial_overlap); act-backend inplace_fix.rs and cpp/malloc constraint.cc; act-oracle load-before-store in the generated code; casestudies (AMX tdp*, PSUM accumulation, Gemmini); RVV register-overlap rules; NKI and PTX operand rules.

### Q9.2: How are those declarations written?

- Options: A. dedicated markers (current or renamed, possibly extended); B. constraint expressions over regions; C. region identity for aliasing plus constraints for the rest; D. a mix, with a stated rule for which is which.
- Evidence: as Q9.1, plus spec-draft S5.4 and S5.5 and panel-2 Q2.2.

### Q9.3: What is the defined outcome when a run violates a declaration, and when regions overlap and the ISA specification declares nothing?

- Options for an undeclared overlap: A. the ISA specification is rejected at generation when overlap is possible; B. a fault at run time; C. a stated default relation; D. other.
- Constraint: owner decisions 16 and 31 together: defined, and not failing scenarios real architectures allow.

### Q9.4: What does each generator do with the declarations, and how does the test corpus check them?

- Context: the oracle must compute the declared value under each relation; the backend must allocate and insert copies accordingly; the corpus needs cases per relation.
- Evidence: panel-4 Q4.6, panel-5 Q5.6, panel-6 Q6.2 decisions.
