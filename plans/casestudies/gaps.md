# Gap matrix

Ranked by severity, then by how many case studies need the construct.
Severity: High means a common instruction needs a workaround that is lossy or depends on implementation-defined behavior; Medium means a workaround works but is verbose or explodes the instruction count; Low means rare or out of scope.
Section ids are those of the spec in `taidl/docs` (S1 to S10).

| Rank | Missing construct | Accelerators needing it | Severity |
| --- | --- | --- | --- |
| 1 | Typed views of untyped storage: one data model read as several element types (S4.3, S7.2) | TPU, Trainium2, Hopper, AMX, SME, AIE-ML, Tensix | High |
| 2 | Strided and multi-dimensional access patterns in a region, for reads and writes (S5.2) | TPU, Trainium2, Hopper, AMX, AIE-ML, Gemmini | High |
| 3 | Body-taking operations: `convolution`, `reduce_window`, `scan`, loops (S8.5) | Eyeriss, Trainium2, CGRA, Tensix | High |
| 4 | Block-scaled and further element types: `bfp8`, MX formats, `tf32` (S7.3) | Tensix, Hopper, Trainium2 | High |
| 5 | Mode-dependent semantics: a case on an attribute or state value instead of `select` over all modes (S6, S7) | TPU, Trainium2, AIE-ML, Gemmini, Tensix, Eyeriss, CGRA | Medium |
| 6 | Numeric annotations: accumulation precision of `dot`, accuracy of transcendentals, rounding mode of `convert` (S8.2, S8.4) | TPU, Trainium2, Hopper, AMX, Tensix | Medium |
| 7 | Instruction families: one declaration over an enumerated attribute such as a tile index or shape mode (S5.1) | AMX, Hopper, AIE-ML, Tensix | Medium |
| 8 | Data-dependent addressing: a start or a state update read from a region, as in descriptors and indirect DMA (S5.2, S5.4) | Trainium2, Hopper, AMX | Medium |
| 9 | Concurrency and instances: engine queues, async copies, tile arrays, neighbor memory (S1 scope) | Trainium2, Hopper, AIE-ML, Tensix | Medium |
| 10 | Indexed state: a state array read by an attribute, as `@s.rows[@c.t]` (S4.4, S6.1) | AMX | Medium |
| 11 | Float-typed attributes and state values (S4.4, S5.1) | Trainium2, Gemmini | Medium |
| 12 | Data models sized at run time: vector length, register mode (S4.3) | SME, Tensix | Low |
| 13 | Programmable sub-units: bitstreams, per-lane programs (S8.5) | CGRA, Tensix | Low |
| 14 | Bit operators in expressions, for packed configuration fields (S6.2) | Gemmini | Low |
| 15 | Array constants or index tables, for swizzles (S7.5, S8.4) | Hopper | Low |
| 16 | A defined value or defined error for a read after a mode change, since v3 has no undefined behavior (S1.3) | SME | Low |
| 17 | VLIW bundles: several slots issued as one instruction (S5) | AIE-ML | Low |
| 18 | Random state, for stochastic rounding (S8.5) | Trainium2 | Low |

Every sketch issues an instruction only where its constraints hold, and several use a constraint as a synchronization stand-in (`mbar_wait`, `lock_acq0`); under v3 a violated constraint must be a defined error (S5.3).

Doc gaps seen in two or more case studies:
- No example uses state variables, constraints, or updates (mini_trn.py:1-102, mini_trn_double.py:1-139 declare none).
- No modeling pattern for a hidden latch, a register file, or a storage whose element type varies.
- The conversion rule of `constant` with a backtick expression and a non-`s64` result type is unstated (semantics.md:69).
- Backtick values inside slice and pad lists are implied by semantics.md:79 and confirmed only in the grammar (IDLV2.g4:19-20, IDLV2.g4:25).
- Mixed-type `dot` and the `gather`, `dynamic_slice` attribute forms are given only by reference to StableHLO (operations.md:102-106, operations.md:121).
