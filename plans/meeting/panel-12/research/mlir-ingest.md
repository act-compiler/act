# MLIR ingest evidence brief for panel 12

Scope: facts only, no position, for Q12.1 to Q12.3 of plans/meeting/panel-12/agenda.md.
Each fact ends with the URL it was read from; URLs are plain text, not links.
Pages were read through WebFetch or WebSearch on 2026-11-05.
Where a page was truncated or silent, the gap is listed in section 7.

## 1. MLIR core infrastructure

### 1.1 Dialects
- The MLIR docs index lists core and contributed dialects, including linalg, tensor, memref, bufferization, vector, affine, scf, transform, pdl, pdl_interp and emitc. https://mlir.llvm.org/docs/Dialects/
- Linalg offers structured ops on tensors and buffers, with linalg.generic as the core and named ops (fill, dot, matmul, conv, pooling) on top. https://mlir.llvm.org/docs/Dialects/Linalg/
- linalg.generic carries indexing maps, explicit parallel or reduction iterator types, a region payload, and an optional library_call attribute. https://mlir.llvm.org/docs/Dialects/Linalg/
- Linalg uses destination-passing style: output operands give the result shape and optionally an initial value. https://mlir.llvm.org/docs/Dialects/Linalg/
- The Python OpDSL for Linalg ops is being deprecated in favor of TableGen ODS definitions. https://mlir.llvm.org/docs/Dialects/Linalg/OpDSL/

### 1.2 ODS and TableGen
- ODS defines operations and some data types in a table-driven way; each op is one TableGen record expanded at build time. https://mlir.llvm.org/docs/DefiningDialects/Operations/
- An op record can state arguments, results, regions, successors, traits, summary and description, and hook flags (hasVerifier, hasFolder, hasCanonicalizer). https://mlir.llvm.org/docs/DefiningDialects/Operations/
- ODS generates the op class, an operand adaptor, typed getters, builders and verifiers from operand, attribute and result constraints. https://mlir.llvm.org/docs/DefiningDialects/Operations/
- An assemblyFormat string generates the parser and printer; custom<...> directives delegate to hand-written C++. https://mlir.llvm.org/docs/DefiningDialects/Operations/

### 1.3 PDL and PDLL
- PDL represents rewrite patterns as MLIR IR: a pdl.pattern op with a benefit attribute, a matcher region and a pdl.rewrite. https://mlir.llvm.org/docs/Dialects/PDLOps/
- PDL values are handles to C++ MLIR objects (!pdl.operation, !pdl.value, !pdl.type, !pdl.attribute, !pdl.range). https://mlir.llvm.org/docs/Dialects/PDLOps/
- PDL patterns are lowered to the pdl_interp dialect for execution. https://mlir.llvm.org/docs/Dialects/PDLOps/
- PDLL is a declarative front-end language for PDL patterns, supporting ahead-of-time and just-in-time compilation. https://mlir.llvm.org/docs/PDLL/
- PDLL presents itself as a replacement for TableGen DRR, which it calls hard to extend for multi-result ops, constraints and multi-op replacement. https://mlir.llvm.org/docs/PDLL/
- A PDLL Pattern has a match section and a rewrite section; default benefit is the number of input operations. https://mlir.llvm.org/docs/PDLL/
- PDLL supports native constraints and rewriters imported from C++ or written as inline C++ blocks. https://mlir.llvm.org/docs/PDLL/
- The PDLL page states it "is still under active development" and its designs may change. https://mlir.llvm.org/docs/PDLL/

### 1.4 Rewrite drivers
- The dialect conversion driver converts ops by legality against a conversion target and supports type conversion. https://mlir.llvm.org/docs/PatternRewriter/
- The greedy driver applies the highest-benefit patterns from a worklist until a fixed point or a maximum iteration count. https://mlir.llvm.org/docs/PatternRewriter/
- Greedy entry points are applyPatternsGreedily (region) and applyOpPatternsGreedily (op list), configured by GreedyRewriteConfig. https://mlir.llvm.org/docs/PatternRewriter/
- GreedyRewriteConfig sets traversal order (bottom-up by default), strictness, scope and iteration cap. https://mlir.llvm.org/docs/PatternRewriter/
- The walk driver (walkAndApplyPatterns) makes one post-order pass and does not revisit new or modified ops. https://mlir.llvm.org/docs/PatternRewriter/
- Pattern benefit is static once the pattern is constructed; a driver can apply a custom cost model via PatternApplicator::applyCostModel. https://mlir.llvm.org/docs/PatternRewriter/
- Recursive pattern application is assumed unbounded and fails if detected, unless the pattern calls setHasBoundedRewriteRecursion. https://mlir.llvm.org/docs/PatternRewriter/

### 1.5 Bufferization
- Bufferization converts ops with tensor semantics into ops with memref semantics, usually late in a pipeline. https://mlir.llvm.org/docs/Bufferization/
- One-Shot Bufferize analyzes a whole function using SSA use-def chains, then rewrites, inserting copies where a buffer cannot be reused. https://mlir.llvm.org/docs/Bufferization/
- One-Shot Bufferize does not deallocate; separate ownership-based passes insert deallocations. https://mlir.llvm.org/docs/Bufferization/
- An op becomes bufferizable by implementing BufferizableOpInterface, including through external models. https://mlir.llvm.org/docs/Bufferization/

### 1.6 Transform dialect
- The transform dialect directs fine-grained transformations of payload IR using transform IR; it is not a replacement for passes or patterns. https://mlir.llvm.org/docs/Dialects/Transform/
- Transform values are operation handles, value handles and parameters; consuming a handle invalidates aliasing handles. https://mlir.llvm.org/docs/Dialects/Transform/
- Each transform reports success, silenceable failure or definite failure, and transform.alternatives can recover from silenceable ones. https://mlir.llvm.org/docs/Dialects/Transform/
- Dialects add transform ops through TransformDialectExtension; no interpreter pass ships, clients drive execution (C++ entry point applyTransforms). https://mlir.llvm.org/docs/Dialects/Transform/

### 1.7 Stability
- MLIR bytecode is documented as versioned; the authors "don't plan to ever break compatibility", conditional on dialects staying immutable. https://mlir.llvm.org/docs/BytecodeFormat/
- Dialects opt into versioned bytecode via BytecodeDialectInterface. https://mlir.llvm.org/docs/BytecodeFormat/
- The LLVM developer policy says the C API is "best effort" for stability and the IR textual format has no specific compatibility promises. https://llvm.org/docs/DeveloperPolicy.html
- Triton states "LLVM does not have a stable API", so each Triton version targets one LLVM revision. https://github.com/triton-lang/triton

## 2. StableHLO

### 2.1 Spec
- StableHLO is an operation set for high-level ML operations, intended as a portability layer between frameworks and compilers. https://openxla.org/stablehlo/spec
- The spec has Programs, Ops and Execution sections; ops are "a closed set" with constraints (C1, C2, ...) and examples per op. https://openxla.org/stablehlo/spec
- Value types are tensor, quantized tensor, token, tuple, buffer and future; "Shapes cannot be unranked"; tuples are legacy. https://openxla.org/stablehlo/spec
- Ops are synchronous by default; tokens and after_all order effects. https://openxla.org/stablehlo/spec
- The repo is based on the MHLO dialect and adds serialization and versioning on MLIR bytecode. https://github.com/openxla/stablehlo
- The README names TensorFlow, JAX and PyTorch as producers and XLA and IREE as consumers. https://github.com/openxla/stablehlo
- The status table tracks 118 ops; interpreter is "yes" for about 97 by one hand count, and the page lists real_dynamic_slice and set_dimension_size as "no". https://github.com/openxla/stablehlo/blob/main/docs/status.md

### 2.2 Versioning and compatibility
- Backward guarantee: artifacts from an older libStablehlo keep semantics in a newer one when builds are less than 5 years apart. https://openxla.org/stablehlo/compatibility
- Forward guarantee: artifacts from a newer libStablehlo keep semantics in an older one when builds are less than 2 years apart, excluding newer features. https://openxla.org/stablehlo/compatibility
- Versions are #.#.#; minor bumps on opset or serialization change, patch bumps weekly and do not affect compatibility. https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
- Guarantees cover only portable artifacts; prettyprinted text, raw bytecode, unspecced features, numerical accuracy and C++/C/Python source compatibility are excluded. https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
- VHLO is a versioned dialect that snapshots ops, types and attributes; it is add-only, with one op per semantic version range (for example VHLO_MyOpV1 and VHLO_MyOpV2). https://github.com/openxla/stablehlo/blob/main/docs/vhlo.md
- Forward compatibility downgrades to a target VHLO version on the producer and fails there if a feature cannot be expressed. https://github.com/openxla/stablehlo/blob/main/docs/vhlo.md

### 2.3 Serialization
- StableHLO serializes with the MLIR bytecode format, credited with faster, smaller output and memory-mapped access. https://github.com/openxla/stablehlo/blob/main/docs/bytecode.md
- C++ API: serializePortableArtifact(ModuleOp, targetVersion, os) and deserializePortableArtifact(source, context). https://openxla.org/stablehlo/compatibility
- Python API: serialize_portable_artifact(module, target_version) and deserialize_portable_artifact(context, artifact). https://openxla.org/stablehlo/compatibility

### 2.4 Reference interpreter
- The interpreter is a readability-first implementation run with stablehlo-translate --interpret. https://github.com/openxla/stablehlo/blob/main/docs/reference.md
- Its C++ entry point is eval(func::FuncOp, ArrayRef<Tensor>) returning SmallVector<Tensor>, with per-op eval functions. https://github.com/openxla/stablehlo/blob/main/docs/reference.md
- Tests use stablehlo.constant inputs and check.expect_eq or check.expect_almost_eq; float checks use a fixed tolerance of 0.0001. https://github.com/openxla/stablehlo/blob/main/docs/reference.md
- The page says numerical accuracy is underspecified (issue 1156). https://github.com/openxla/stablehlo/blob/main/docs/reference.md

## 3. torch-mlir

- torch-mlir aims for first-class PyTorch-to-MLIR compiler support and is mainly integrated into other compilers; the README names IREE and BladeDISC downstream. https://github.com/llvm/torch-mlir
- The README names ONNX and FX entry points and says the project is in the LLVM Incubator, not part of any LLVM release. https://github.com/llvm/torch-mlir
- The architecture page describes a frontend producing the torch dialect and a backend contract (value semantics, known rank and dtype, decomposed ops). https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
- The torch dialect maps almost 1:1 to PyTorch JIT IR; most ops are generated from the PyTorch operator registry into ODS. https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
- Lowering targets are Linalg-on-Tensors (most complete, full dynamic shapes), TOSA (static shapes, quantized types) and StableHLO (dynamic shapes). https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md
- The roadmap page (marked last updated 2022Q4) says the TorchScript path is deprecated and the FX/export path "not fully functional yet" at that time. https://github.com/llvm/torch-mlir/blob/main/docs/roadmap.md
- The same roadmap lists the three backends as implemented independently with duplication and suggests torch-mlir may move under the PyTorch umbrella. https://github.com/llvm/torch-mlir/blob/main/docs/roadmap.md
- A 2025 LLVM talk describes the FX importer as taking a torch.export ExportedProgram to the torch dialect, and says Torch-to-Linalg has more coverage than Torch-to-TOSA. https://www.llvm.org/devmtg/2025-04/slides/technical_talk/bartel_upstream.pdf
- An open issue reports FX import of Qwen3-8B taking over 20 minutes versus 2-3 minutes via ONNX. https://github.com/llvm/torch-mlir/issues/4312
- A torch-mlir release 20261101 was uploaded to PyPI for cp310 to cp312 on macOS, Linux and Windows. https://pypi.org/pypi/torch-mlir/json
- Development doc: LLVM is a submodule at externals/llvm-project, updated roughly weekly to a green commit; PyTorch defaults to the latest nightly. https://github.com/llvm/torch-mlir/blob/main/docs/development.md
- Separate path: torch.export to StableHLO exists in torch_xla through exported_program_to_stablehlo, with optional dynamic_shapes. https://docs.pytorch.org/xla/master/features/stablehlo.html
- iree-turbine registers as a torch.compile backend (TorchDynamo capture) and offers aot.export producing MLIR then a .vmfb. https://iree.dev/guides/ml-frameworks/pytorch/

## 4. IREE

### 4.1 HAL
- The HAL dialect is "a Vulkan-like model with all of the graphics bits chopped out", covering buffers, buffer views, synchronization and command buffers. https://iree.dev/reference/mlir-dialects/HAL/
- HAL op groups include hal.device.*, hal.executable.* (create, lookup, export, variant), hal.command_buffer.* (dispatch, copy_buffer, execution_barrier) and hal.allocator.* / hal.buffer.*. https://iree.dev/reference/mlir-dialects/HAL/
- The design roadmap says the HAL is modeled nearly one-to-one on compute-only Vulkan so compiler passes can plan allocation, lifetime and synchronization. https://iree.dev/developers/design-docs/design-roadmap/
- The HIP driver doc maps iree_hal_executable_t to hipModule_t; the CUDA doc says the HAL layer is expected to be "a thin layer". https://iree.dev/developers/design-docs/hip-hal-driver

### 4.2 Plugin points for a custom accelerator
- The IREE compiler is "a monolithic shared object with a dynamic plugin system", and a plugin adds some combination of target backends, options, passes or pipelines. https://iree.dev/reference/bindings/c-api/
- Plugins connect through a Plugin API with static or dynamic linking; the page points to compiler/PluginAPI/README.md. https://iree.dev/reference/bindings/c-api/
- In-tree HAL target plugins live in compiler/plugins/target and use shared code in compiler/src/iree/compiler/Dialect/HAL/Target/. https://opensecura.googlesource.com/3p/openxla/iree/+show/HEAD/compiler/plugins/target/README.md
- Static plugins are selected with IREE_COMPILER_PLUGINS at configure time. https://iree.dev/developers/building/bazel/
- Dynamic compiler plugins are labeled experimental; they must be built against the host compiler's IREE and LLVM/MLIR revisions with matching ABI settings. https://iree.dev/developers/building/bazel/
- The CMake options page lists IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS and loading via --iree-load-plugin=<path>. https://iree.dev/developers/building/cmake-options/
- Input dialect options are IREE_INPUT_STABLEHLO, IREE_INPUT_TORCH and IREE_INPUT_TOSA. https://iree.dev/developers/building/cmake-options/
- Out-of-tree example: iree-amd-aie is built by pointing IREE_CMAKE_PLUGIN_PATHS at its root and adding a runtime driver via IREE_EXTERNAL_HAL_DRIVERS=amdxdna. https://github.com/nod-ai/iree-amd-aie
- That repo ships iree_compiler_plugin.cmake and iree_runtime_plugin.cmake, and calls itself "early-phase". https://github.com/nod-ai/iree-amd-aie

## 5. Dependency and build cost

- StableHLO builds against one LLVM commit recorded in build_tools/llvm_version.txt; the README says to rebuild MLIR whenever it changes. https://github.com/openxla/stablehlo
- The README cites about 10 minutes for the MLIR build on an M1 Pro. https://github.com/openxla/stablehlo
- The StableHLO pin file content read on 2026-11-05 was commit c744694c6ae1d12bbf7d0447b4bc91ee94a1904c. https://raw.githubusercontent.com/openxla/stablehlo/main/build_tools/llvm_version.txt
- torch-mlir pins LLVM as a submodule; an out-of-tree build needs LLVM built with -DLLVM_ENABLE_PROJECTS=mlir at a matching commit. https://github.com/llvm/torch-mlir/blob/main/docs/development.md
- Triton pins LLVM in cmake/llvm-info.json (llvm_hash), downloads a prebuilt LLVM by default, and says a source LLVM build "will take a while". https://github.com/triton-lang/triton
- IREE states builds took "around 5-10 minutes on 64-core Linux machines" in 2023 and calls the compiler build complex. https://iree.dev/building-from-source/getting-started/
- IREE can drop the compiler (-DIREE_BUILD_COMPILER=OFF) and restrict targets with IREE_TARGET_BACKEND_DEFAULTS=OFF and IREE_HAL_DRIVER_DEFAULTS=OFF. https://iree.dev/building-from-source/getting-started/
- IREE allows a bring-your-own LLVM copy when embedded in other projects. https://iree.dev/building-from-source/getting-started/
- IREE keeps patches to llvm-project in forks and bumps the submodule with a script (older mirrored doc). https://opensecura.googlesource.com/3p/openxla/iree/+/5e2d790b411a4477421c4fcae5cf3dc03f252a77/docs/developers/developing_iree/llvm_version_bump.md
- MLIR getting-started: build with LLVM_ENABLE_PROJECTS=mlir; restrict LLVM_TARGETS_TO_BUILD, use clang, lld and ccache for speed; split DWARF saves roughly 30-40 percent disk. https://mlir.llvm.org/getting_started/
- The MLIR page gives no memory, disk total or version-pinning guidance. https://mlir.llvm.org/getting_started/
- Binary size, packaged: iree-base-compiler 3.13.0 Linux x86_64 wheels are about 83.6 million bytes each. https://pypi.org/pypi/iree-base-compiler/json
- Binary size, packaged: torch-mlir 20261101 manylinux x86_64 wheels are about 79.5 million bytes. https://pypi.org/pypi/torch-mlir/json
- Binary size, packaged: an LLVM 22.1.4 MLIR runtime-libs package lists about 147 MB installed on one distro. https://pkgs.chimera-linux.org/package/current/main/ppc/mlir-libs

## 6. How existing accelerator flows define their target ISA in MLIR

### 6.1 Buddy-mlir Gemmini
- A gemmini dialect is defined in TableGen at midend/include/Dialect/Gemmini/Gemmini.td. http://buddy.ruyiai.org/Pages/Documentation/Gemmini.html
- Ops mirror ISA instructions: flush, config_st, config_ld, config_ex, config_norm, mvin, mvin2, mvin3, mvout, preload, compute_preloaded, compute_accumulated, plus tile_matmul and tile_conv. http://buddy.ruyiai.org/Pages/Documentation/Gemmini.html
- Intrinsic-level ops include loop_ws and loop_conv_ws with config variants, mapped to LLVM intrinsics defined in IntrinsicsRISCVBuddyExt.td. http://buddy.ruyiai.org/Pages/Documentation/Gemmini.html
- Lowering: linalg.matmul to gemmini.tile_matmul, linalg conv ops to gemmini.tile_conv, a legalize pass keeps only intrinsic ops, and LLVM IR translation is table-generated. http://buddy.ruyiai.org/Pages/Documentation/Gemmini.html
- The hardware ISA is RoCC custom RISC-V instructions (mvin, mvout, config_*, matmul.preload, matmul.compute), not in GNU binutils; C macros build encodings. https://github.com/ucb-bar/gemmini
- Merlin (IREE-based) defines a different gemmini dialect of "recovery" ops: clamp, conv2d, matmul, matmul_tile, requantize. https://ucb-bar.github.io/merlin/reference/generated/mlir/gemmini_dialect
- The Merlin log says its gemmini-lower-to-isa pass only does a structural matmul-to-matmul_tile step and nothing is validated on hardware. https://ucb-bar.github.io/merlin/dev_blog/2026-03-11-gemmini-workstream-log

### 6.2 mlir-aie
- mlir-aie is an MLIR toolchain for AMD AI Engine arrays, with IRON Python API and multiple IR levels; it links against LLVM and builds it from source. https://github.com/Xilinx/mlir-aie
- Peano extends LLVM with the AI Engine processor as a target, for AIE2 and AIE2P. https://github.com/Xilinx/mlir-aie
- The aie dialect describes a device as a netlist: aie.device (with a device attribute), aie.tile(col, row), aie.core, aie.buffer, aie.lock, aie.use_lock, aie.switchbox, aie.connect, aie.flow. https://xilinx.github.io/mlir-aie/latest/AIEDialect/
- Data movement is modeled by aie.dma, aie.dma_bd, aie.dma_start and aie.next_bd. https://xilinx.github.io/mlir-aie/latest/AIEDialect/
- The aie.core region holds code that is outlined to the LLVM dialect and compiled to a binary per core. https://xilinx.github.io/mlir-aie/latest/AIEDialect/

### 6.3 Triton
- Triton v2.0 backend was rewritten to use MLIR; third-party backend code lives under third_party/. https://github.com/triton-lang/triton
- Lowering runs Triton IR (TTIR) to TritonGPU IR (TTGIR) with layouts (blocked, shared) to LLVM IR and PTX or AMDGCN. https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/
- Triton 3.7 plugin extensions load shared libraries from TRITON_PLUGIN_PATHS and add passes, separately compiled dialects and conversion passes. https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/
- The post lists dynamically loaded out-of-tree backends as future work, not shipped. https://pytorch.org/blog/triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box/

### 6.4 NVDLA
- A search for an NVDLA MLIR dialect or compiler flow returned no specific result. https://arxiv.org/pdf/2202.03293

## 7. Gaps and unverified items

- No page read gave an MLIR build time in minutes for a stock LLVM tree; only StableHLO (10 minutes, M1 Pro) and IREE (5-10 minutes, 64 cores) quotes exist.
- No page read gave a stripped binary size for a linked MLIR-based tool; the sizes above are packaged wheels or distro packages.
- The torch-mlir roadmap page is dated 2022Q4 and may be stale; the development doc was read without a date.
- The IREE plugin guide (compiler/PluginAPI/README.md) was not retrieved; only secondary descriptions were read.
- The Merlin Gemmini facts come from a project blog and a dialect reference; they were read as fetched, not checked against code.
- The mlir-aie AIE dialect page was truncated at 100000 of about 283000 characters; the AIEDevice enum values were not seen.
- The StableHLO spec page was truncated; the total op count and the Execution section were not read, and the 118-op figure comes from the status table.
- No source was found for an MLIR C++ API stability promise; only the LLVM developer policy C API statement and the Triton quote above.
