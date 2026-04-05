# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Multi-stage Container Image

# ── Stage 1: Build Rust FFI extension ────────────────────────────
# Pin base images by digest for reproducible builds
FROM rust:1.83-slim@sha256:89e45d1b4de96d61e457618ae4da44690eae5578fe2f11d26d1ed02ce5c8e412 AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/ci-tools.txt /tmp/ci-tools.txt
RUN pip3 install --break-system-packages --no-cache-dir \
    --require-hashes --no-deps -r /tmp/ci-tools.txt

WORKDIR /build
COPY spo-kernel/ spo-kernel/

RUN cd spo-kernel && \
    maturin build --release -m crates/spo-ffi/Cargo.toml --out /wheels

# ── Stage 2: Build Python package ────────────────────────────────
FROM python:3.12-slim@sha256:2be8daddbd3438e0e0c82ddd4a37e0e7ff3c1e0a0e7e0e4ed4e3be0ba26d3e21 AS python-builder

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/
COPY domainpacks/ domainpacks/
COPY --from=rust-builder /wheels/*.whl /wheels/

RUN pip install --no-cache-dir --prefix=/install . /wheels/*.whl

# ── Stage 3: Production image ────────────────────────────────────
FROM python:3.12-slim@sha256:2be8daddbd3438e0e0c82ddd4a37e0e7ff3c1e0a0e7e0e4ed4e3be0ba26d3e21 AS production

LABEL maintainer="Miroslav Sotek <protoscience@anulum.li>"
LABEL org.opencontainers.image.source="https://github.com/anulum/scpn-phase-orchestrator"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"

RUN groupadd --gid 1000 spo && \
    useradd --uid 1000 --gid spo --create-home spo

COPY --from=python-builder /install /usr/local
COPY --chown=spo:spo domainpacks/ /app/domainpacks/

WORKDIR /app
USER spo

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; r=u.urlopen('http://localhost:8000/api/health'); assert b'healthy' in r.read()"]

ENTRYPOINT ["spo"]
CMD ["--help"]
