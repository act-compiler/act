FROM amd64/ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive

# Base dependencies for ACT backend compilation
RUN apt-get update && \
    apt-get install -y \
        wget curl git \
        cmake make ninja-build \
        g++ clang-14 gcc \
        python3 python3-pip python3-dev \
        zlib1g-dev libabsl-dev lsb-release \
        graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install \
    numpy==1.26.4 \
    scipy==1.13.1 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    pandas==2.2.1 \
    jax==0.4.33 \
    nnsmith==0.1.0 \
    tensorflow==2.16.2 \
    antlr4-python3-runtime==4.9.2 && pip3 cache purge

# Install Rust
ENV RUSTUP_HOME=/root/.rustup
ENV CARGO_HOME=/root/.cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install OR-Tools (for Phase 2 CSP solver)
ENV ORTOOLS_DIR=/opt/ortools
RUN wget -q https://github.com/google/or-tools/releases/download/v9.12/or-tools_amd64_ubuntu-22.04_cpp_v9.12.4544.tar.gz -O /tmp/ortools.tar.gz && \
    mkdir -p $ORTOOLS_DIR && \
    tar -xzf /tmp/ortools.tar.gz -C $ORTOOLS_DIR --strip-components=1 && \
    rm -rf /tmp/ortools.tar.gz
ENV LD_LIBRARY_PATH="$ORTOOLS_DIR/lib"

# Install nlohmann/json (for C++ backend)
ENV NLOHMANN_JSON_DIR=/opt/nlohmann_json
RUN wget -q https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz -O /tmp/json.tar.xz && \
    mkdir -p $NLOHMANN_JSON_DIR && \
    tar -xf /tmp/json.tar.xz -C $NLOHMANN_JSON_DIR --strip-components=1 && \
    rm -rf /tmp/json.tar.xz

# Symlink python/pip to python3/pip3
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

WORKDIR /act

CMD ["bash"]
