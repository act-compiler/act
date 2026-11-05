# Joint moderator decision: reading on-chip storage that nothing wrote

Binding inputs: decisions 16, 31, 40; panel 1 Q1.4, Q1.9, Q1.12; panel 4 Q4.9.

| Question | Decision | Confidence | Owner? |
| -------- | -------- | ---------- | ------ |
| Unwritten read | The Q1.4 fault stands as the default; an ISA specification may declare a per-buffer initial value, and a read of an unwritten element of such a buffer yields that value | medium-high | no |

## The norm

- One sentence: "Before the first issue an on-chip buffer holds its declared initial value in every element, or no value; a read of an element that holds no value and that no earlier issue of the program wrote is a fault."
- The initial value is optional per buffer, a literal of the element type under the Q1.10 literal rules; `d0` holds the caller's `hbm` in full and takes none.
- "Written" means written by an issue of this program; a stale value left by an earlier program is not detected, as S5.6 already says.
- The fault keeps the Q1.4 and Q1.12 shape: decided on host integers before any read, writes no region and no state, named by the lowest issue index.

## Reasoning chain

- 1. Real hardware splits in two: AMX zeroes every tile on `ldtilecfg` and SME zeroes ZA when `PSTATE.ZA` goes from 0 to 1, so those architectures define the initial contents; NKI states that `accumulate=True` on an uninitialized PSUM location is undefined behavior on all targets.
- 2. Decision 31 protects scenarios real architectures allow; a scenario the vendor itself names undefined behavior is not one the architecture allows, so the fault on undeclared storage does not contradict 31.
- 3. Decision 31 is contradicted only where the architecture does define the contents: a kernel that accumulates into a freshly configured AMX tile is legal on hardware, and under Q1.4 as worded it faults; this is the genuine tension the oracle author found, and the declared initial value is its spelling.
- 4. Decision 16 closes the other routes for undeclared storage: an unspecified value is undefined behavior by another name, and a silent zero is the fiction panel 1 rejected, since an oracle pass would certify the NKI kernel the vendor warns about.
- 5. The declaration follows the principle of decision 31 as the joint 1-5 decision applied it: the default never makes a hardware-legal program invalid, and the ISA author states the hardware's behavior where it differs from the default.
- 6. Decision 40 is met: the fault depends only on region text, attributes, and state (Q1.9), exactly as the out-of-bounds region fault of S1.3 does, so it is a property of the assembly code and not of tensor data; the plans already read every S1.3 fault as "invalid assembly code" (oracle decisions.md failure policy section, diagnostics D14).
- 7. The cause is genuine and correctable: the program reads storage its architecture leaves undefined, and the kernel author fixes it with the ISA's zeroing or overwriting first issue (`tilezero`, `zero za`, `accumulate=False`).
- 8. The backend side of panel 4 Q4.9 and B36 stands for undeclared buffers; for a buffer with a declared value a read before write is legal and the backend may rely on the value, which lets an AMX backend omit `tilezero` where the architecture already zeroed.
- 9. Oracle cost is the Q1.4 boolean mask for undeclared buffers only; a declared buffer keeps a prologue fill, which is today's `sh_zeros` path with the declared literal in place of zero, so no new emitter is needed.
- 10. A declaration is a small language addition in the shape of `add_state(name, value)`, which already gives a state variable its value before the first issue; it belongs to the language plan next to L32 and needs no new concept.

## Evidence

- Verified: act/plans/language/spec-draft/execution.md:16-22 (storage before the first issue, S1.3.11 fault, "written" during this program).
- Verified: act/plans/language/spec-draft/declarations.md:36 (`add_state(name, value)` declares the value before the first issue).
- Verified: act-oracle/taidl_to/templates/templates.txt:469-470 (`PROLOGUE_INIT` emits `sh_zeros` per buffer).
- Verified: act/plans/oracle/plan/checks.md:83 and act/plans/oracle/plan/decisions.md:78 (the author's flag against decisions 31 and 40).
- Verified: act/plans/backend/plan/extractor.md:40-47 (B36 drop reason, no zeroing); act/plans/meeting/panel-4/decision.md:119-121 (Q4.9).
- Verified: panel 1 round1 and round2 files, Q1.4 sections (all three debaters on option C; no dissent).
- Fetched: Intel SDM `LDTILECFG`, "Tiles are zeroed in the INIT state" and `zero_all_tile_data()` on a valid load.
- Fetched: NKI `nki.isa.nc_matmul`, "Using accumulate=True on an uninitialized PSUM location is undefined behavior on all targets."
- Searched, not the normative text: Linux KVM SME series, "setting PSTATE.ZA to 1 causes ZA and ZT0 to be reset to 0"; the Arm ARM wording was not fetched.

## Dissent and weight

- The oracle author: real hardware lets the read happen, so a fault fails an allowed scenario; right for AMX and SME, answered by the declaration, and wrong for NKI, where the read is not an allowed scenario.
- Panel 1 (all three): a plain fault with no declaration; it stands where no value is declared, and the declaration removes the one case where it contradicted decision 31.

Confidence: medium-high.
Would change the ruling: an in-scope architecture whose scratchpad is neither defined at kernel start nor documented undefined, and whose kernels read it by design.
