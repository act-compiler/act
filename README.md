# ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions

This is the artifact for our paper on automatically generating sound and complete compiler backends for tensor accelerators from their ISA descriptions written in TAIDL.

This artifact consists of ACT source code, TAIDL ISA definitions, input kernel benchmarks, and the necessary scripts to reproduce the evaluation results.
To facilitate artifact evaluation, we have automated the entire environment setup and experimental processes as part of Docker images.

## System Requirements

**Docker**

We use Docker images for environment setup of ACT and baselines.
To run the ACT artifact, install Docker using the [installation guide](https://docs.docker.com/engine/install/).

**Architecture Support**

- **amd64/x86_64** is required. arm64 is not currently supported.

## ACT Artifact Overview

### Scripts

`scripts/` directory contains the following bash scripts to automate the experimental workflows:

- `setup.sh` -- load the necessary Docker images from local tarballs or Docker Hub.
- `exec.sh` -- execute a given script inside the Docker container.
- `launch.sh` -- launch an interactive Docker shell for development.
- `clean.sh` -- clean all generated files for a fresh start.

### Docker images

| Docker image                    | Environment Installed        | Architecture | Dockerfile                  |
| ------------------------------- | ---------------------------- | ------------ | --------------------------- |
| `devanshdvj/act-artifact:amd64` | ACT (Rust, OR-Tools, Python) | amd64        | `./artifact-act/Dockerfile` |

## Step-by-step Instructions to Reproduce Results

### (Step 0) Setup the Docker Environment

This will first try to load the necessary Docker images from tarballs locally present in `./` or `../` (i.e., current or parent directory).

If the tarballs are not found locally, it will try to pull the images from Docker Hub.

This may take 1-2 minutes if loading from local tarballs, or 4-5 minutes if pulling from Docker Hub.

Run the setup script using:

```bash
./scripts/setup.sh
```

Alternatively, you can build the Docker images locally using:

```bash
./scripts/setup.sh --build
```

### (Step 1) QKV

```bash
./scripts/exec.sh artifact-act/run-qkv.sh
```

### (Optional) Generate fuzzing HLO Kernels using NNSmith

This is optional since the fuzz-generated kernels are already included in the artifact.

```bash
./scripts/exec.sh kernels/setup/nnsmith/setup_nnsmith.sh
```

### (Step 2) Gemmini

```bash
./scripts/exec.sh artifact-act/run-gemmini.sh
```

### (Optional) Interactive Environment

To start an interactive Docker shell for development:

```bash
./scripts/launch.sh
```

The artifact is mounted at `/act/` in the Docker container.

### (Optional) Cleanup/Fresh Start

To remove all generated files including generated compiler backends, compiled kernels, and generated plots for a fresh start:

```bash
./scripts/clean.sh
```

## Project Structure

- `accelerators/` - TAIDL ISA definitions
- `kernels/` - Input HLO benchmark programs
  - `qkv/` - QKV kernels
  - `gemmini/` - Gemmini kernels (SW library + fuzz-generated)
- `artifact-act/` - ACT Docker environment and scripts to run ACT experiments
- `plots/` - Visualization scripts
- `scripts/` - Automation scripts
