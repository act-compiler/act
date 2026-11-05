# Ways to attach an accelerator backend to PyTorch

Evidence brief for panel 12, Q12.2; it takes no position.
Each fact ends with a source key that resolves to a URL in the Sources table; versions are as of PyTorch 2.15 [rel].

## Summary

The table restates facts cited in the sections below.

| Path | Entry point | IR the backend receives | Unsupported operators | Coupling to PyTorch releases |
| --- | --- | --- | --- | --- |
| torch.compile backend | `backend=` callable or registered name | FX graph from Dynamo; core ATen after AOTAutograd | graph break runs in eager Python | private `torch._dynamo` imports |
| torch.export | `torch.export.export` then `run_decompositions` | ExportedProgram; core ATen ~180 ops | export errors out | core ATen opset still growing |
| PrivateUse1 device | C++ kernels on the `PrivateUse1` dispatch key | single ATen ops in eager | registered CPU fallback | C++ ABI; stable ABI is limited |
| ExecuTorch delegate | `partition` + `preprocess` + runtime `init`/`execute` | Edge dialect subgraphs | runs on the base ExecuTorch runtime | pins a torch nightly |
| torch-mlir | `fx.export_and_import`, `stateless_fx_import` | torch dialect, then Linalg, TOSA, or StableHLO | not covered in its docs | pins a torch nightly |
| PJRT via PyTorch/XLA | PJRT C API plugin + `DevicePlugin` | XLA/StableHLO programs | ops not lowered run on CPU | project deprecated |

## torch.compile custom backends

- Entry point: a function `(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable` passed as `torch.compile(..., backend=fn)` [cb].
- Registration by `@register_backend` or a `torch_dynamo_backends` entry point enables `backend="name"` and the minifier [cb].
- Backends wrapped by `torch._dynamo.backends.common.aot_autograd` receive forward and backward graphs through `fw_compiler` and `bw_compiler` [cb].
- The docs state that AOTAutograd produces FX graphs of core ATen ops, a much smaller opset than all of ATen [cb].
- Callables returned under AOTAutograd must be "boxed", for example via `functorch.compile.make_boxed_func` [cb].
- Decompositions can be supplied as a table; torch-mlir's `export_and_import` takes a `decomposition_table` and calls `run_decompositions` with it [tmfx].
- Guards: Dynamo records assumptions about locals and globals as guards and recompiles on a guard failure, up to `torch._dynamo.config.recompile_limit` [faq].
- Dynamic shapes: the default "automatic dynamic" compiles static first, then marks changed dimensions dynamic on recompile [dyn].
- `torch._dynamo.mark_dynamic(tensor, dim, min=, max=)` marks a dimension dynamic up front and fails if it gets specialized [dyn].
- Fallback: on an untraceable construct, `torch.compile` graph-breaks and runs that part in eager Python; `fullgraph=True` disables this [exp] [faq].
- `torch._dynamo.config.suppress_errors = True` skips a function whose compile crashes; the docs call this "not best practice" [trb].
- Testing: the troubleshooting guide offers `TORCHDYNAMO_REPRO_AFTER="dynamo"` or `"aot"` and `TORCHDYNAMO_REPRO_LEVEL=4` for accuracy [trb].
- The same guide states that "most of the time nowadays, the minifier doesn't work as expected" [trb].
- Maintenance: the documented AOTAutograd hook lives under the private `torch._dynamo` namespace [cb].
- PyTorch ships a minor release roughly every two months; 2.13 to 2.16 fall on 8 Jul, 2 Sep, 28 Oct, and 22 Dec 2026 [rel].

## torch.export and its IR

- Entry point: `torch.export.export(module, args, dynamic_shapes=...)` returns an `ExportedProgram` [exp].
- Unlike `torch.compile`, export "will error out when something untraceable is reached" rather than graph-break [exp].
- Dialects: the default Training IR has 3000+ ATen ops including mutation [exp].
- `run_decompositions(decomp_table={})` gives a functional Inference IR of ~2000 ops [exp].
- `run_decompositions(decomp_table=None)` gives core ATen IR of ~180 ops, described as optimal for backends minimizing operator count [exp].
- `torch.export.default_decompositions()` returns an editable table, so a backend can keep or redefine decompositions per op [exp].
- The 2.15 IR page lists 193 core ATen ops and 123 prims ops [ir].
- Core ATen IR is fully functional, with no in-place or `_out` variants [ir].
- Stability: both opsets carry the warning "still under active development, more ops will be added in the future" [ir].
- ExecuTorch says core ATen "was developed with stability in mind" yet "cannot be considered fully complete" until all ATen is classified [etops].
- Dynamic shapes use `torch.export.Dim("name", min=, max=)`, `Dim.DYNAMIC`, or `Dim.AUTO` [exp].
- Serialization: `torch.export.save` and `load` write a zip following the PT2 Archive Spec [exp] [pt2].
- CPU fallback: none inside export; partial lowering belongs to the consumer, as ExecuTorch delegates do (below) [etdel].

## PrivateUse1 out-of-tree device (eager)

- Entry point: register C++ kernels for the `PrivateUse1` dispatch key with `TORCH_LIBRARY_IMPL`, plus runtime pieces [accops] [pu1].
- PyTorch now recommends PrivateUse1 for new backends because each dedicated key touched many files and keys are capped at 64 [pu1].
- The official guide calls it the default integration path for future PyTorch features and uses `torch_openreg` as reference [acc].
- Runtime work: device management, accelerator hooks, guard, generator, streams, events, and memory [acc] [pu1].
- The minimum set is factory, view, and copy ops: `empty.memory_format`, `empty_strided`, `as_strided`, `view`, `resize_`, `_copy_from`, `_local_scalar_dense`, and `set_` variants [accops].
- Fallback: `m.fallback` with `at::native::cpu_fallback` sends unimplemented ops to CPU; per-op and blacklist variants exist [accops].
- `torch.utils.rename_privateuse1_backend` names the device, and an `_autoload` entry point loads it at `import torch` [accauto].
- For `torch.compile`, an `_inductor_backend_init` hook calls `register_backend_for_device` at the first Inductor compile [accauto].
- Testing: OOT backends run upstream ACCELERATOR tests via `PYTORCH_TESTING_DEVICE_FOR_CUSTOM`, skipping missing ops by config [acctest].
- CI: the Cross-Repository CI Relay forwards PyTorch PR events downstream; of levels L1 to L4, only L1 (silent) works today [accci].
- Maintenance: the CI chapter states that upstream changes "can potentially break downstream accelerator integrations" [accci].
- The LibTorch Stable ABI covers custom operators and extensions, with headers promised unmodified for at least 2 years [sabi].
- That ABI is a "limited" interface; a kernel is version-independent only if it uses only that subset [sabi].

## ExecuTorch delegates

- Entry point: a delegate takes Edge dialect, produced by `to_edge`, from a torch.export program [etdel].
- Ahead of time, `partition(exported_program) -> PartitionResult` tags nodes, and `to_backend` lowers each tagged subgraph [etdel].
- `preprocess(edge_program, compile_specs) -> PreprocessResult` returns a binary blob serialized into the `.pte` file [etdel].
- At runtime the backend implements `is_available`, `init`, `execute`, and optionally `destroy` [etdel].
- Fallback: untagged parts run on the basic ExecuTorch runtime [etcon].
- Testing: `backends/test/suite` runs backend-agnostic operator and model tests per registered flow, reporting delegated versus undelegated passes [etsuite].
- Maintenance: ExecuTorch marks APIs Experimental, Stable, or Deprecated; Stable APIs change only after a deprecation period [etapi].
- ExecuTorch main pins `TORCH_VERSION = "2.14.0"` with nightly `dev20261004` [etpin].

## torch-mlir

- Entry points: `torch_mlir.fx.export_and_import` imports an `ExportedProgram`; `stateless_fx_import` imports a bare FX graph [tmfx].
- `export_and_import` accepts `dynamic_shapes` and `decomposition_table`; `OutputType` options are TORCH, LINALG_ON_TENSORS, TOSA, STABLEHLO, and RAW [tmfx].
- The "backend contract" guarantees value semantics, known rank, and known dtype, plus configurable decompositions [tmarch].
- Built-in lowerings are Linalg-on-tensors, TOSA, and StableHLO [tmarch].
- The StableHLO lowering "functionally supports dynamic shapes", though less coherently than Linalg [tmarch].
- Testing: an end-to-end suite under `projects/pt1/python/torch_mlir_e2e_test/` runs per config, for example `fx_importer_tosa` [tmdev].
- Maintenance: torch-mlir is an LLVM incubator project, not part of any LLVM release [tmrd].
- It pins `torch==2.15.0.dev20260917` [tmpin].
- CPU fallback for unsupported ops is not described in its README or architecture doc [tmrd] [tmarch].

## PyTorch/XLA and PJRT plugins

- Status: the PyTorch/XLA README marks the project deprecated in favor of TorchTPU, with the repo archived on 15 Nov 2026 [xlard].
- Entry point: a binary implementing `PjRtClient` behind the C `PJRT_Api`, plus a Python `DevicePlugin` exposed via a `torch_xla.plugins` entry point [xlaplug].
- The same PJRT binary can serve JAX and TensorFlow [xlaplug].
- The `DevicePlugin` API is labeled experimental [xlaplug].
- The PJRT C API header is at major version 0, minor 117 [pjrth].
- StableHLO promises 5 years of backward and 2 years of forward compatibility for portable artifacts [shlo].
- Fallback: ops without an XLA lowering run on CPU, visible as `aten::` counters in metrics reports [xlatrb].
- Testing: a plugin can be tried before packaging with `PJRT_DEVICE=LIBRARY PJRT_LIBRARY_PATH=...` [xlaplug].

## How existing accelerators attach

The table restates the cited bullets below; the ROCm version is from [rel].

| Vendor | Eager path | Compile path | IR to vendor compiler | Release pairing |
| --- | --- | --- | --- | --- |
| Google TPU (TorchTPU) | PrivateUse1, three eager modes | Dynamo to XLA, `backend="tpu"` | StableHLO | not stated |
| AWS Neuron (TorchNeuron) | PrivateUse1, per-op NEFF | Dynamo + AOTAutograd, `backend="neuron"` | StableHLO via vendored torch_mlir | `torch-neuronx==2.14.*` for PyTorch 2.14 |
| Intel Gaudi | HPU device | `backend="hpu_backend"` | bridge graph | own fork or public 2.13 (preview) |
| Tenstorrent | not covered here | PyTorch/XLA into PJRT | StableHLO to TT-MLIR | not stated |
| Cerebras | none (lazy capture) | `cstorch.compile` | ATen dialect to CIRH | not stated |
| Meta MTIA | `MTIA` key in-tree, impl out-of-tree | Inductor with Triton | Triton | not stated |
| AMD ROCm | in-tree, reuses `torch.cuda` | not covered here | not stated | ROCm 7.14 listed for 2.15 |

- TorchTPU uses PrivateUse1 with "no subclasses, no wrappers", and compiles Dynamo graphs through XLA rather than Inductor [tpublog].
- TorchTPU lists reduced recompilation under dynamic shapes as an open challenge [tpublog].
- TorchNeuron, public beta in Neuron 2.33, registers via PrivateUse1, autoloads, and validates forward and backward against upstream ATen test suites [nab].
- TorchNeuron eager runs unsupported ops on CPU with a warning [nab].
- TorchNeuron `torch.compile` requires `dynamic=False`, and an op it cannot lower fails compilation by default [nrn] [nab].
- Gaudi eager mode "falls back to the CPU for any op that is not supported", then resumes on Gaudi [gth].
- Gaudi 1.24 supports PyTorch 2.13 and `torch.export` in eager plus compile mode, but not `backend="inductor"` [gmx] [gsm].
- Tenstorrent ingests PyTorch through PyTorch/XLA into its PJRT plugin, and marks the torch-mlir-based TT-Torch deprecated [ttx].
- Cerebras built on the XLA lazy tensor backend, passing the ATen graph through to its MLIR IR, CIRH [cblog].
- `cstorch.compile` only prepares the model; tracing a step yields the graph compiled for CSX [cpkg].
- PyTorch docs state "The MTIA backend is implemented out of the tree, only interfaces are defined here" [mtia].
- Meta reports a Triton backend for MTIA-2i with TorchInductor codegen changes, in production across about 60 model types [mtiapaper].
- PyTorch for HIP "intentionally reuses the existing torch.cuda interfaces" [hip].
- Intel GPU (XPU) support is ready from PyTorch 2.5 as a prototype, with eager and `torch.compile` [xpu].

## Sources

| Key | URL |
| --- | --- |
| acc | https://docs.pytorch.org/docs/2.15/accelerator/index.html |
| accauto | https://docs.pytorch.org/docs/2.15/accelerator/autoload.html |
| accci | https://docs.pytorch.org/docs/2.15/accelerator/ci.html |
| accops | https://docs.pytorch.org/docs/2.15/accelerator/operators.html |
| acctest | https://docs.pytorch.org/docs/main/accelerator/testing.html |
| cb | https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_custom_backends.html |
| cblog | https://www.cerebras.ai/blog/supporting-pytorch-on-the-cerebras-wafer-scale-engine |
| cpkg | https://training-docs.cerebras.ai/rel-2.8.0/cs-torch/cerebras-pytorch-api/cerebras-pytorch-package |
| dyn | https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html |
| etapi | https://docs.pytorch.org/executorch/stable/api-life-cycle.html |
| etcon | https://docs.pytorch.org/executorch/stable/concepts.html |
| etdel | https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html |
| etops | https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html |
| etpin | https://github.com/pytorch/executorch/blob/main/torch_pin.py |
| etsuite | https://github.com/pytorch/executorch/blob/main/backends/test/suite/README.md |
| exp | https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export.html |
| faq | https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_faq.html |
| gmx | https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html |
| gsm | https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Support_Matrix.html |
| gth | https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html |
| hip | https://docs.pytorch.org/docs/2.15/notes/hip.html |
| ir | https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_ir.html |
| mtia | https://docs.pytorch.org/docs/2.15/mtia.html |
| mtiapaper | https://arxiv.org/abs/2608.00325 |
| nab | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/torch/about-pytorch-neuron.rst |
| nrn | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/release-notes/components/torchneuron.rst |
| pjrth | https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h |
| pt2 | https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/pt2_archive.html |
| pu1 | https://docs.pytorch.org/tutorials/advanced/privateuseone.html |
| rel | https://github.com/pytorch/pytorch/blob/main/RELEASE.md |
| sabi | https://docs.pytorch.org/docs/2.15/notes/libtorch_stable_abi.html |
| shlo | https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md |
| tmarch | https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md |
| tmdev | https://github.com/llvm/torch-mlir/blob/main/docs/development.md |
| tmfx | https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/fx.py |
| tmpin | https://github.com/llvm/torch-mlir/blob/main/pytorch-requirements.txt |
| tmrd | https://github.com/llvm/torch-mlir/blob/main/README.md |
| tpublog | https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/ |
| trb | https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html |
| ttx | https://github.com/tenstorrent/tt-xla/blob/main/README.md |
| xlaplug | https://github.com/pytorch/xla/blob/master/docs/source/contribute/plugins.md |
| xlard | https://github.com/pytorch/xla/blob/master/README.md |
| xlatrb | https://github.com/pytorch/xla/blob/master/docs/source/learn/troubleshoot.md |
| xpu | https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html |
