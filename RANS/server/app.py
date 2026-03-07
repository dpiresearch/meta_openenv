# Copyright (c) Space Robotics Lab, SnT, University of Luxembourg, SpaceR
# RANS: arXiv:2310.07393 — OpenEnv-compatible implementation

"""
FastAPI application for the RANS spacecraft navigation environment.

Exposes the RANSEnvironment over HTTP/WebSocket using OpenEnv's
``create_app`` factory, following the same pattern as echo_env and coding_env.

Usage
-----
Development (auto-reload)::

    uvicorn rans_env.server.app:app --reload --host 0.0.0.0 --port 8000

Production::

    uvicorn rans_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

Docker entry-point::

    uv run --project . server

Task selection (default: GoToPosition)::

    RANS_TASK=GoToPose uvicorn rans_env.server.app:app --host 0.0.0.0 --port 8000
"""

from rans_env.models import SpacecraftAction, SpacecraftObservation
from rans_env.server.rans_environment import RANSEnvironment

try:
    from openenv.core.env_server import create_app
except ImportError as exc:
    raise RuntimeError(
        "openenv-core is required to run the RANS server.  "
        "Install it with:  pip install openenv-core"
    ) from exc

# Pass the class (not an instance) so create_app can spin up one environment
# per WebSocket session, enabling concurrent independent episodes.
app = create_app(
    RANSEnvironment,
    SpacecraftAction,
    SpacecraftObservation,
    env_name="rans_env",
)


def main() -> None:
    """Entry-point for ``uv run --project . server``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
