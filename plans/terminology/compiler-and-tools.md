# Compiler backend, tools, and errors

"Current" names the term in `taidl/docs/glossary.md` first, then other repo usage.

## Tools and artifacts

| Concept | Current TAIDL/ACT term(s) | Owner papers | Other tools | Recommended term | One-line rationale |
| --- | --- | --- | --- | --- | --- |
| Tool that reads an ISA description | front end (`taidl` package) | none | frontend | front end | Keep. |
| Tool generated from an ISA description | generator (TAIDL-TO, ACT-Backend) | "compiler backend generator" [ACT] | n/a | generator | Keep. |
| Generated compiler | compiler backend | "compiler backend" [ACT] | backend (XLA, MLIR) | compiler backend | Owner's term; matches every tool. |
| Executable model of the ISA | test oracle; TAIDL-TO; modes `fsim`, `fsim-compile` | "test oracles (also known as functional simulators)" [TAIDL] | `nki.simulate_kernel`, NKI CPU simulator [NKI] | test oracle | Owner's title term; gloss "functional simulator" on first use. |
| Input unit of compilation | kernel; "tensor kernels in XLA-HLO IR" (`act-backend/README.md:16`) | "tiled kernels (sub-graphs with typically 10-50 nodes)" 48x [ACT] | `gpu.func` with `kernel` attribute [MLIR]; `@nki.jit` kernel [NKI] | tiled kernel | Owner's precise term for the XLA-HLO input. |
| Unit run on the test oracle | kernel; `@kernel` function of instruction calls (`act-oracle/README.md:37-50`) | "kernel" [TAIDL] | `@nki.jit` kernel [NKI] | kernel | Same word as NKI for a function of ISA calls. |
| Compiled output | kernel (glossary bans "assembly"); "machine code" (`act-backend/README.md:16,19`); `assembly/` (`act-oracle/README.md:46`) | "assembly code ... a stream of concrete instructions and a constant tensor" [ACT] | n/a | assembly code -- CONFLICT | Owner defines it exactly; the glossary bans "assembly". |
| Source IR | XLA-HLO IR | "XLA-HLO IR, XLA's tensor IR" [ACT] | HLO [XLA] | XLA-HLO | Name the IR exactly. |

## Compiler backend pieces

| Concept | Current term(s) | Owner papers | Other tools | Recommended term | One-line rationale |
| --- | --- | --- | --- | --- | --- |
| Phase 1 | none in repos | "instruction selection" using equality saturation [ACT] | lowering | instruction selection | Standard compiler term, used by [ACT]. |
| Program set | "e-graph nodes" (`act-backend/README.md:63`) | e-graph, e-class, e-node [ACT] | n/a | e-graph, e-class, e-node | Owner's terms. |
| Rewrite | "rewrites" (`act-backend/README.md:62-63`) | "IR-to-IR" and "IR-to-ISA rewrite rules" [ACT] | `RewritePattern` [MLIR] | rewrite rule | "rewrite" alone also means regenerating goldens in that README. |
| Choosing a program | "extractors" (`act-backend/README.md:65`) | "extraction", "graph extractor" [ACT] | n/a | extraction, extractor | Owner's terms. |
| Partial instruction plan | none | "partially instantiated instruction (pii)", "pii graph" [ACT] | n/a | partially instantiated instruction | Owner's term; define it once. |
| Phase 2 | "allocator" (`act-backend/README.md:110`) | "memory allocation using integer constraint programming" [ACT] | bufferization, allocation [MLIR] | memory allocation | Owner's term; matches common usage. |
| Backend correctness | soundness and completeness (`act-backend/README.md:19`) | "sound and complete" [ACT] | n/a | sound, complete | Owner's terms. |

## Errors and conformance

| Concept | Current term(s) | Other tools | Recommended term | One-line rationale |
| --- | --- | --- | --- | --- |
| Raising an error for a description | reject (glossary); "refuse", "refuses", "refused" in `act-oracle/README.md`, `act-oracle/docs/operations.md`, `act-backend/README.md:48`, `taidl/docs/{semantics,expressions,declarations,operations}.md`, `taidl/taidl/{instruction,names}.py` | violated constraint (Cn) [SHLO]; legality (dialect conversion) [MLIR] | reject | The glossary chose it; the repos have not followed. |
| Valid but not handled | unsupported feature; "supports, rejects or supports in part" | legality (dialect conversion) [MLIR] | unsupported feature | Keep; a generator rejects an unsupported feature. |
| Meeting the language | conforming front end, conforming generator (`taidl/docs/overview.md` S1.1, S1.2) | n/a | conforming, conformance | Standard language-spec usage. |
| Result not given | undefined behavior | "the behavior is undefined" [SHLO] | undefined behavior | Same as [SHLO]. |
| Left to each tool | implementation-defined | implementation-defined [SHLO] | implementation-defined | Same as [SHLO]. |
