# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — On-demand simulation server command

"""Run a simulation API in the foreground with explicit native admission."""

from __future__ import annotations

from pathlib import Path

import click

from scpn_phase_orchestrator.runtime.cli._app import main


@main.command(help="Serve a binding's simulation in the foreground; stop with Ctrl-C.")
@click.argument(
    "spec_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=8000, show_default=True)
@click.option(
    "--require-kernel/--allow-python",
    default=True,
    show_default=True,
    help="Require verified native computation before serving requests.",
)
def serve(spec_path: Path, host: str, port: int, require_kernel: bool) -> None:
    """Serve a binding's simulation until interrupted or terminated.

    Parameters
    ----------
    spec_path : pathlib.Path
        Binding specification for the in-process simulation.
    host : str
        Listening address; defaults to host loopback.
    port : int
        TCP port in the range 1 through 65535.
    require_kernel : bool
        Refuse unavailable native computation unless Python is explicitly allowed.

    Raises
    ------
    click.ClickException
        A server dependency, binding or required native kernel is unusable.
    """
    try:
        import uvicorn

        from scpn_phase_orchestrator.runtime.server import create_app

        app = create_app(spec_path, require_kernel=require_kernel)
    except (ImportError, RuntimeError, ValueError, AssertionError) as exc:
        raise click.ClickException(str(exc)) from exc
    uvicorn.run(app, host=host, port=port, access_log=True)
