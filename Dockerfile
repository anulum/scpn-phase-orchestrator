# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multi-stage Container Image

# ── Stage 1: Build Rust FFI extension ────────────────────────────
# Pin base images by digest for reproducible builds. Build the FFI wheel with
# the same CPython minor as the runtime image so the extracted extension imports.
FROM python:3.13-slim@sha256:e544a7fcbdf8555eceda66bf86cafb006c736339f76141918bcb812f3174c00a AS rust-builder

# rustup 1.29.1 installer; digest published beside rustup-init upstream.
ENV CARGO_BUILD_JOBS=1 \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH \
    RUSTUP_INIT_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -fsSL \
        https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
        -o /tmp/rustup-init && \
    echo "${RUSTUP_INIT_SHA256}  /tmp/rustup-init" | sha256sum -c - && \
    chmod +x /tmp/rustup-init && \
    /tmp/rustup-init -y --profile minimal --default-toolchain 1.95.0 && \
    rm /tmp/rustup-init

COPY requirements/ci-tools.txt /tmp/ci-tools.txt
RUN python -m pip install --no-cache-dir \
    --require-hashes --no-deps -r /tmp/ci-tools.txt

WORKDIR /build
COPY spo-kernel/ spo-kernel/

RUN cd spo-kernel && \
    maturin build --release -m crates/spo-ffi/Cargo.toml --out /wheels

# ── Stage 2: Build Python package ────────────────────────────────
FROM python:3.13-slim@sha256:e544a7fcbdf8555eceda66bf86cafb006c736339f76141918bcb812f3174c00a AS python-builder

WORKDIR /build

COPY requirements/build-tools.txt /tmp/build-tools.txt
RUN python -m pip install --no-cache-dir \
    --require-hashes --no-deps -r /tmp/build-tools.txt
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY requirements/server-lock.txt /tmp/server-lock.txt
COPY --from=rust-builder /wheels/*.whl /wheels/

# The project is built into a wheel with the hash-pinned build tools above, so
# the install step consumes only local wheels and no unpinned source tree.
RUN python -m pip install --no-cache-dir --prefix=/install \
        --require-hashes --no-deps -r /tmp/server-lock.txt && \
    python -m pip wheel --no-cache-dir --no-deps --no-build-isolation \
        --wheel-dir /wheels . && \
    python -m pip install --no-cache-dir --prefix=/install \
        --no-deps /wheels/*.whl

# ── Stage 3: Production image ────────────────────────────────────
FROM python:3.13-slim@sha256:e544a7fcbdf8555eceda66bf86cafb006c736339f76141918bcb812f3174c00a AS production

LABEL maintainer="Miroslav Sotek <protoscience@anulum.li>"
LABEL org.opencontainers.image.source="https://github.com/anulum/scpn-phase-orchestrator"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

ARG SECURITY_REFRESH_STAMP=manual
RUN echo "security refresh ${SECURITY_REFRESH_STAMP}" >/dev/null && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 spo && \
    useradd --uid 1000 --gid spo --create-home spo

COPY --from=python-builder /install /usr/local
COPY --chown=spo:spo domainpacks/ /app/domainpacks/

WORKDIR /app
USER spo

# Plain HTTP is deliberate: this is an in-container loopback liveness probe; it
# crosses no trust boundary and carries no credentials or sensitive payload.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import json, urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4); assert json.load(r)['status']=='healthy'"]

ENTRYPOINT ["python", "-c", "from scpn_phase_orchestrator.runtime.cli import main; main()"]
EXPOSE 8000
CMD ["serve", "domainpacks/minimal_domain/binding_spec.yaml", "--host", "0.0.0.0", "--port", "8000", "--require-kernel"]
