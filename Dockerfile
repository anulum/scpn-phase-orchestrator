# SCPN Phase Orchestrator
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

COPY . .

ENTRYPOINT ["spo"]
