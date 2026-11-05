# Gap matrix

Ranked by severity, then by the number of public sites.
Severity: High means a common instruction needs a workaround that is lossy or cannot be stated; Medium means a workaround works but is verbose or explodes the instruction count; Low means rare or out of scope.
Section ids are those of the TAIDL language specification (S1 to S12).

## Counting rule

- A site is a public, documented ISA in owner-15 scope; the invented Eyeriss and CGRA command sets count zero.
- A site counts toward admission only after its sketch ran through the front end; no sketch has run, so every count below is documented sites and the ran count is 0 until C11.
- An extension needs an idiom failure (the idiom loses information or cannot be stated; verbosity is not failure) plus two ran public sites (Q2.3).
- The L-track lead writes each count; a reviewer outside the track checks each cited document and run.

## Matrix

| Rank | Missing construct | Public sites | Severity | Item | v3 decision |
| --- | --- | --- | --- | --- | --- |
| 1 | Strided and multi-dimensional access patterns in a region (S5.2) | TPU, Trainium2, Hopper TMA, AMX, AIE-ML, Gemmini | High | C2, L19 | Covering-range idiom loses information; L19 is first after v3, admitted only if the C2 footprint demonstration shows the idiom fails |
| 2 | Element types `f4e2m1fn`, `f8e8m0fnu` (S7.3) | Blackwell, NeuronCore-v4 (Trainium3) | High | C6, L17 | Admitted in v3; no `tf32` |
| 3 | Accumulation precision of `dot` (S8.2) | TPU, Trainium2, Hopper, AMX, Tensix | Medium | C7 | Cannot be stated; waits for a second ran site |
| 4 | Data-dependent addressing: descriptors read at run time, state updates read from data (S5.2, S5.4) | Trainium2 DGE, Hopper tensor map | Medium | C8, L23 | One study with L23; no new state kind |
| 5 | Typed views of untyped storage (S4.3, S7.2) | TPU, Trainium2, Hopper, AMX, SME, AIE-ML, Tensix | Medium | C1 | Idiom `u8` plus `bitcast_convert`; byte order fixed by F6; no syntax |
| 6 | Body-taking operations: `convolution`, `scan` (S8.5) | Trainium2 `tensor_tensor_scan`, Tensix; NVDLA queued | Medium | C5, L20 | Idiom; L22 cut |
| 7 | Mode-dependent semantics (S6, S7) | TPU, Trainium2, AIE-ML, Gemmini, Tensix | Medium | C4 | `select` over modes; one failure case, the `wgmma` transpose flags |
| 8 | Instruction families over an enumerated attribute (S5.1) | AMX, Hopper, AIE-ML, Tensix | Medium | C3 | Python loop in the ISA specification |
| 9 | Block-scaled dot and `bfp8` (S7.3) | Tensix, Trainium2, Blackwell | Medium | C6, L17 | `scaled_dot` stays an idiom; exact for power-of-two scales |
| 10 | Indexed state: `@s.rows[@c.t]` (S4.4, S6.1) | AMX | Medium | C3 | One site; Hopper `mbarrier` arrays count only with a state kind, which v3 does not add |
| 11 | Concurrency and instances (S1) | Trainium2, Hopper, AIE-ML, Tensix | Medium | C10 | Out of v3; program order is the only order |
| 12 | Float-typed attributes and state (S4.4, S5.1) | Trainium2, Gemmini | Low | C9 | The `u32` bits route is lossless; not admitted |
| 13 | Bit operators for packed fields (S6.2) | AMX, Hopper, Gemmini | Low | C9 | `/` and `%` unpack exactly; not admitted |
| 14 | Aliased buffers over one storage | Tensix `Dst` | Low | C10 | One site |
| 15 | Programmable sub-units (S8.5) | Tensix | Low | C10 | One site; CGRA is invented |
| 16 | Array constants for swizzles (S7.5, S8.4) | Hopper | Low | C10 | One site; `iota` plus `gather` idiom |
| 17 | VLIW bundles (S5) | AIE-ML | Low | C10 | One site |
| 18 | Random state (S8.5) | Trainium2 | Low | C10 | One site |

## Notes

- Rank 1 of the earlier matrix (typed views) is Medium: the only lossy part was the implementation-defined byte order, which F6 fixes.
- The three corrected verdicts: `ldtilecfg` is expressible as is, indirect DMA is expressible with a workaround, and `smstop` is a defined fault.
- Indexed state (rank 10) rests on `tileloadd` and `tdpbf16ps` only.
- Several sketches use a constraint as a wait (`mbar_wait`, `wgmma_wait_group`, `lock_acq0`); under v3 a false constraint faults (S5.3), sound only for program order (C13).

Doc gaps seen in two or more case studies:
- No example uses state variables, constraints, or updates (mini_trn.py:1-102, mini_trn_double.py:1-139 declare none).
- No modeling pattern for a hidden latch, a register file, or a storage whose element type varies.
- The conversion rule of `constant` with a backtick expression and a non-`s64` result type is unstated (semantics.md:69).
- Backtick values inside slice and pad lists are implied by semantics.md:79 and confirmed only in the grammar (IDLV2.g4:19-20, IDLV2.g4:25).
- Mixed-type `dot` and the `gather`, `dynamic_slice` attribute forms are given only by reference to StableHLO (operations.md:102-106, operations.md:121); C7 depends on the mixed-type form.
