# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multi-stage Container Image

# ── Stage 1: Build Rust FFI extension ────────────────────────────
# Pin base images by digest for reproducible builds. Build the FFI wheel with
# the same CPython minor as the runtime image so the extracted extension imports.
FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d AS rust-builder

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH \
    RUSTUP_INIT_SHA256=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10

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
FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d AS python-builder

WORKDIR /build

COPY requirements/runtime-lock.txt /tmp/runtime-lock.txt
COPY --from=rust-builder /wheels/*.whl /wheels/

RUN python -m pip install --no-cache-dir --prefix=/install \
        --require-hashes --no-deps -r /tmp/runtime-lock.txt && \
    python -c "import glob, pathlib, sysconfig, zipfile; target = pathlib.Path(sysconfig.get_paths(vars={'base': '/install', 'platbase': '/install'})['platlib']); target.mkdir(parents=True, exist_ok=True); wheels = glob.glob('/wheels/*.whl'); assert len(wheels) == 1, wheels; zipfile.ZipFile(wheels[0]).extractall(target)"

# ── Stage 3: Production image ────────────────────────────────────
FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d AS production

LABEL maintainer="Miroslav Sotek <protoscience@anulum.li>"
LABEL org.opencontainers.image.source="https://github.com/anulum/scpn-phase-orchestrator"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

RUN groupadd --gid 1000 spo && \
    useradd --uid 1000 --gid spo --create-home spo

COPY --from=python-builder /install /usr/local
COPY --chown=spo:spo src/ /app/src/
COPY --chown=spo:spo domainpacks/ /app/domainpacks/

WORKDIR /app
ENV PYTHONPATH=/app/src
USER spo

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; r=u.urlopen('http://localhost:8000/api/health'); assert b'healthy' in r.read()"]

ENTRYPOINT ["python", "-c", "from scpn_phase_orchestrator.cli import main; main()"]
CMD ["--help"]
