# CPU build of the engine, plus the GPT-2 weights, in one image.
#
#   docker build -t gpt2-engine .
#   docker run --rm -p 8080:8080 gpt2-engine
#   curl -N localhost:8080/v1/generate -d '{"prompt": "The meaning of life is", "max_tokens": 40, "stream": true}'
#
# Stages:
#   toolchain  compilers, cargo and LibTorch; also a dev shell (see README, "Build with Docker")
#   build      compiles the engine and runs the test suite
#   weights    downloads GPT-2 from Hugging Face and exports it (scripts/export_weights.py)
#   runtime    the binaries, LibTorch's shared libraries and the weights; serves on port 8080

ARG LIBTORCH_VERSION=2.10.0

FROM ubuntu:24.04 AS toolchain
ARG LIBTORCH_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
RUN curl -fL --retry 3 -o /tmp/libtorch.zip \
        "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip" \
    && unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV LIBTORCH_DIR=/opt/libtorch


FROM toolchain AS build
WORKDIR /src
COPY . .
# Compiling against the LibTorch headers takes ~2 GB per job.
ARG JOBS=4
RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch \
    && cmake --build /build -j "${JOBS}" \
    && ctest --test-dir /build --output-on-failure


FROM python:3.12-slim AS weights
ARG LIBTORCH_VERSION
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${LIBTORCH_VERSION}" \
    && pip install --no-cache-dir "transformers>=5" safetensors
COPY scripts/export_weights.py /tmp/
RUN python /tmp/export_weights.py --out /weights && rm -rf /root/.cache/huggingface


FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/libtorch/lib/ /opt/libtorch/lib/
COPY --from=build /build/gpt2_serve /build/gpt2_generate /build/gpt2_loadgen /build/gpt2_bench /usr/local/bin/
COPY --from=weights /weights/ /app/weights/
ENV LD_LIBRARY_PATH=/opt/libtorch/lib
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fs localhost:8080/health || exit 1
ENTRYPOINT ["gpt2_serve", "--weights", "/app/weights", "--host", "0.0.0.0", "--port", "8080"]
