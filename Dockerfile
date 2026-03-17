# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Container Image

FROM python:3.12-slim

RUN groupadd --gid 1000 spo && useradd --uid 1000 --gid spo --create-home spo

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir . && chown -R spo:spo /app

COPY --chown=spo:spo . .

USER spo

ENTRYPOINT ["spo"]
