# ACT: Accelerator Compiler Toolkit

Recent years have seen a proliferation of specialized ML accelerators – proposed in both academia (e.g., Gemmini, FEATHER) and industry (e.g., Google TPU, Intel AMX) -- that depart significantly from traditional CPU/GPU architectures. However, research on compiler and systems support for these accelerators remains sparse, largely due to the lack of mature open-source compiler infrastructures capable of targeting them from popular ML frameworks like PyTorch, and JAX. Building such support involves considerable manual effort, slowing innovation and creating a gap between hardware and software research communities.

To bridge this gap, we present **ACT (Accelerator Compiler Toolkit) Ecosystem**, an ecosystem that automatically generates complete compiler backends and essential software tooling from high-level ISA specifications of tensor accelerators.

The ACT ecosystem consists of:

- **TAIDL (Tensor Accelerator ISA Definition Language)**: A Python-based DSL for specifying tensor accelerator ISAs. TAIDL leverages tensor IRs like XLA-HLO to compactly and precisely model execution semantics of tensor accelerator ISAs.
- **TAIDL-TO (Test Oracle) Generator**: Automatically generates scalable functional simulators just from TAIDL specifications, enabling correctness testing of the software stack. TAIDL-TOs are orders of magnitude faster than existing simulators.
- **ACT Backend Generator**: Automatically generates sound and complete compiler backends just from TAIDL specification. ACT backends match or outperform state-of-the-art expert-written libraries, while maintaining low compile times (<1 sec).
- **XLA Integration**: Enables end-to-end compilation from popular ML frameworks like JAX, TensorFlow, and PyTorch

## Repositories

The ACT Ecosystem is composed of multiple repositories:

- [act-compiler/act](https://github.com/act-compiler/act): Top-level repository for the ACT Ecosystem, containing submodules for relevant tools.
- [act-compiler/taidl](https://github.com/act-compiler/taidl): Repository for TAIDL, the Tensor Accelerator ISA Definition Language.  
  Mounted as a submodule at `./taidl/`.
- [act-compiler/act-oracle](https://github.com/act-compiler/act-oracle): Repository for TAIDL-TO, the Test Oracle generator.  
  Mounted as a submodule at `./generators/oracle/`.
- [act-compiler/act-backend](https://github.com/act-compiler/act-backend): Repository for ACT Backend, the Compiler Backend generator.  
  Mounted as a submodule at `./generators/backend/`.

## Interested in using ACT?

If you are interested in using ACT for your accelerator, please email [Devansh Jain](mailto:devansh9@illinois.edu) or [Prof. Charith Mendis](mailto:charithm@illinois.edu).
We would be happy to help you get started!

## Interested in contributing to ACT?

If you are interested in contributing to the ACT Ecosystem (e.g., adding new language features, adding new tool generators, improving existing generators, etc.), please feel free to open an issue or a pull request in the relevant repository.

| Contribution Area          | Repository Link                                                                |
| -------------------------- | ------------------------------------------------------------------------------ |
| TAIDL Language Features    | [act-compiler/taidl](https://github.com/act-compiler/taidl)                    |
| New Tool Generators        | [act-compiler/act](https://github.com/act-compiler/act) (top-level repository) |
| Test Oracle Generator      | [act-compiler/act-oracle](https://github.com/act-compiler/act-oracle)          |
| Compiler Backend Generator | [act-compiler/act-backend](https://github.com/act-compiler/act-backend)        |
| Documentation & Bug Fixes  | Corresponding repository                                                       |

**Unable to decide?** Open an issue in the top-level repository: [act-compiler/act](https://github.com/act-compiler/act) and we can help route it to the appropriate place.

# Citing ACT

If ACT helps you in your academic research, you are encouraged to cite our papers.

For TAIDL and TAIDL-TO, please cite our MICRO 2025 paper:

```
@inproceedings{taidl-micro2025,
  author = {Jain, Devansh and Frigo, Marco and Arora, Jai and Pardeshi, Akash and Wang, Zhihao and Patel, Krut and Mendis, Charith},
  title = {TAIDL: Tensor Accelerator ISA Definition Language with Auto-generation of Scalable Test Oracles},
  year = {2025},
  isbn = {9798400715730},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3725843.3756075},
  doi = {10.1145/3725843.3756075},
  booktitle = {Proceedings of the 2025 58th IEEE/ACM International Symposium on Microarchitecture},
  pages = {1316–1333},
  numpages = {18},
  series = {MICRO '25},
  month = oct
}
```

For ACT Backend, please cite our arXiv preprint:

```
@misc{act-arxiv,
  title = {ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions},
  author = {Jain, Devansh and Pardeshi, Akash and Frigo, Marco and Patel, Krut and Khulbe, Kaustubh and Arora, Jai and Mendis, Charith},
  year = {2025},
  eprint = {2510.09932},
  archiveprefix = {arXiv},
  primaryclass = {cs.PL},
  url = {https://arxiv.org/abs/2510.09932},
  doi = {10.48550/arXiv.2510.09932},
  month = oct
}
```

# Acknowledgements

This project was supported in part by ACE, one of the seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program sponsored by DARPA; and by NSF under grant CCF-2338739.
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Government.
