# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - ClusterFuzzLite builder image

FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:a7652a04e2db56911868c9aeab9fa4eb1dbabb6a9e45772b2fa718dd4471fa25

COPY . "$SRC/scpn-phase-orchestrator"
WORKDIR "$SRC/scpn-phase-orchestrator"

RUN python3 -m pip install --require-hashes --no-deps -r requirements/runtime-lock.txt
