# Issue log: casestudies-priorart

Source is `reviews/casestudies-priorart/review.md` (CP1 to CP20).
The case-study and prior-art tracks have no review-response file; each row was checked against `casestudies/`, `priorart/`, and panel 2.
Severity is the review's own label.
Panel questions are cited as Qn.k and owner decisions as ODn.

| Finding | Severity | Disposition | Where it landed |
| --- | --- | --- | --- |
| CP1: the P2 taxonomy keeps "undefined" as a word | major | fixed | OD16, Q1.12; priorart/plan/P2-taxonomy.md (rejected, fault, defined value) |
| CP2: `ldtilecfg` is expressible as one instruction | major | fixed | Q2.7; casestudies/gaps.md (notes, rank 10 rests on `tileloadd` and `tdpbf16ps`); casestudies/amx.md |
| CP3: indirect DMA is expressible with a workaround | major | fixed | Q2.7; casestudies/gaps.md (rank 4); casestudies/plan/c08_data_dependent_addressing.md |
| CP4: C2, C5, C6, C7, C8 duplicate L19 to L23 | major | fixed | casestudies/plan/README.md (an item that duplicates an L item points to it); Q2.3 |
| CP5: R59 recount per construct, with a decision column | major | fixed | Q2.3; casestudies/gaps.md (counting rule, v3 decision column) |
| CP6: C8 and P11 give two solutions for descriptors | major | superseded by Q2.5: no new state kind in v3, data-dependent addressing is one study joined with L23 | casestudies/plan/c08_data_dependent_addressing.md |
| CP7: no descriptor-driven engine study | major | fixed | Q2.7; casestudies/plan/c12_rerun_on_v3.md (NVDLA queued) |
| CP8: rank 1 rests on a byte order that OD16 removes | minor | fixed | casestudies/gaps.md (typed views at Medium); casestudies/plan/c01_typed_views.md; F6 |
| CP9: invented Eyeriss and CGRA command sets count as evidence | minor | fixed | casestudies/gaps.md (invented sets count zero) |
| CP10: C5 admits `reduce` with a custom reducer and `reduce_window` | minor | fixed | casestudies/plan/c05_body_operations.md (scope: bounded `scan`) |
| CP11: `tf32` is not a storage type | minor | fixed | Q2.4; casestudies/plan/c06_element_types.md |
| CP12: `smstart` zeroing and `smstop` fault | minor | fixed | Q2.7; casestudies/sme.md; casestudies/gaps.md (notes) |
| CP13: `wgmma` sketch omits the immediates and the async group | minor | fixed | casestudies/hopper.md (`wgmma_64x64x16` attributes, `wgmma_commit_group`) |
| CP14: a constraint used as a wait faults under R55 | minor | fixed | Q2.6; casestudies/plan/c13_defined_behavior_audit.md; casestudies/gaps.md (notes) |
| CP15: shape bugs in the sketches (`slice` rank, `copy`) | minor | fixed | casestudies/hopper.md (`reshape` per line); casestudies/plan/c11_executable_corpus.md |
| CP16: Tensix `Dst` is aliased storage, SVL is not a gap | minor | fixed | casestudies/gaps.md (rank 14, aliased buffers) |
| CP17: P7 and C11 boundary unstated | minor | fixed | casestudies/plan/c11_executable_corpus.md; priorart/plan/P7-examples-as-tests.md |
| CP18: stride has two homes, P11 and C2 | minor | fixed | priorart/plan/P11 (stride belongs to C2 and L19) |
| CP19: P3, P4, P6, P10, P12 are backend refactors; move P6 after v3 | minor | fixed | priorart/plan/README.md (P6 after v3) |
| CP20: mixed-type `dot` given only by reference | minor | fixed | casestudies/plan/c10_modeling_cookbook.md (doc gap); casestudies/plan/c07_numeric_annotations.md |
