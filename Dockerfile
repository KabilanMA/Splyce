# syntax=docker/dockerfile:1
#
# Builds Splyce against the pinned LLVM/MLIR commit from README.md
# (Building and Installation, Step 1 Option A) and produces an image that
# runs the README's Usage Examples out of the box. Doesn't bundle
# experiments/ (the Evaluation harness) yet.
#
# Build: docker build -t splyce .
#   Build args: LLVM_COMMIT, NINJA_JOBS (cap ninja's parallelism — the
#   link steps are memory-hungry), GCC_MAJOR_VERSION (must be available
#   in the ubuntu:24.04 repos).
#
# Run: docker run --rm -it splyce
#   Paste any Usage Examples command — WORKDIR is /splyce, and
#   mlir-opt/mlir-translate/clang/splyce-opt/splyce-translate are on $PATH.

########################################################################
# Stage 1: build LLVM/MLIR/clang/lld/openmp from source, pinned to the
# commit this project is tested against.
########################################################################
FROM ubuntu:24.04 AS llvm-builder

ARG LLVM_COMMIT=6a6d432550598a59605ee062bd0e35c9d452c0c5
ARG GCC_MAJOR_VERSION=13
ARG LLVM_INSTALL_PREFIX=/opt/llvm-install
ARG NINJA_JOBS=

# GCC_INSTALL_DIR is derived from GCC_MAJOR_VERSION rather than its own
# ARG, so it can't drift out of sync with the gcc/g++ packages below. ENV,
# not ARG, so both survive into stage 2 without re-specifying.
ENV GCC_INSTALL_DIR=/usr/lib/gcc/x86_64-linux-gnu/${GCC_MAJOR_VERSION} \
    LLVM_INSTALL=${LLVM_INSTALL_PREFIX}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build git python3 "gcc-${GCC_MAJOR_VERSION}" "g++-${GCC_MAJOR_VERSION}" zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Shallow-fetch just the pinned commit instead of the full history.
WORKDIR /src/llvm-project
RUN git init . \
    && git remote add origin https://github.com/llvm/llvm-project.git \
    && git fetch --depth 1 origin "$LLVM_COMMIT" \
    && git checkout FETCH_HEAD

# Mirrors README.md Step 1 Option A, except CMAKE_C/CXX_COMPILER name the
# versioned gcc-${GCC_MAJOR_VERSION}/g++-${GCC_MAJOR_VERSION} explicitly
# (the only variant installed above).
RUN cmake -S llvm -B build -G Ninja \
      -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir;lld;openmp" \
      -DLLVM_ENABLE_RUNTIMES="all" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_TARGETS_TO_BUILD="host;X86" \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLLVM_USE_LINKER=bfd \
      -DCMAKE_C_COMPILER="gcc-${GCC_MAJOR_VERSION}" \
      -DCMAKE_CXX_COMPILER="g++-${GCC_MAJOR_VERSION}" \
      -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
      -DRUNTIMES_CMAKE_ARGS="-DCMAKE_C_FLAGS=--gcc-install-dir=$GCC_INSTALL_DIR;-DCMAKE_CXX_FLAGS=--gcc-install-dir=$GCC_INSTALL_DIR" \
    && ninja -C build ${NINJA_JOBS:+-j$NINJA_JOBS} install \
    && rm -rf /src/llvm-project/build

RUN "$LLVM_INSTALL/bin/mlir-opt" --version

########################################################################
# Stage 2: configure/build Splyce itself against the LLVM install from
# stage 1 (README.md "Building and Installation" -> Step 2).
########################################################################
FROM llvm-builder AS splyce-builder

ARG NINJA_JOBS=

ENV PATH="${LLVM_INSTALL}/bin:${PATH}"

WORKDIR /splyce
COPY CMakeLists.txt ./
COPY include ./include
COPY lib ./lib
COPY tools ./tools
COPY playground ./playground

RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DMLIR_DIR="${LLVM_INSTALL}/lib/cmake/mlir" \
      -DLLVM_DIR="${LLVM_INSTALL}/lib/cmake/llvm" \
      -DCMAKE_C_FLAGS="--gcc-install-dir=${GCC_INSTALL_DIR}" \
      -DCMAKE_CXX_FLAGS="--gcc-install-dir=${GCC_INSTALL_DIR}" \
    && ninja -C build ${NINJA_JOBS:+-j$NINJA_JOBS}

# README.md "Building and Installation" -> Step 3: Verify Build.
RUN ./build/bin/splyce-opt --help | grep splyce

########################################################################
# Stage 3: slim runtime image for README.md's Usage Examples — the LLVM
# install, splyce-opt/splyce-translate, and playground/'s inputs.
########################################################################
FROM ubuntu:24.04 AS runtime

ARG GCC_MAJOR_VERSION=13

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      "gcc-${GCC_MAJOR_VERSION}" "g++-${GCC_MAJOR_VERSION}" python3 zlib1g ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_INSTALL=/opt/llvm-install
ENV PATH="${LLVM_INSTALL}/bin:/splyce/build/bin:${PATH}"

COPY --from=llvm-builder /opt/llvm-install /opt/llvm-install

WORKDIR /splyce
COPY --from=splyce-builder /splyce/build/bin/splyce-opt /splyce/build/bin/splyce-translate ./build/bin/
COPY --from=splyce-builder /splyce/playground ./playground

# Fail the build here rather than ship a broken image.
RUN ./build/bin/splyce-opt --help | grep splyce \
    && mlir-opt --version \
    && clang --version

CMD ["/bin/bash"]
