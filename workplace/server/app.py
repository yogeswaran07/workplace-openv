"""FastAPI application for the workplace policy OpenEnv server."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required for the web server. Install with "
        "'pip install openenv-core[core]'."
    ) from exc

try:
    from ..models import WorkplaceAction, WorkplaceObservation
    from .workplace_environment import WorkplaceEnvironment
except ImportError:  # pragma: no cover - direct repo execution path
    from models import WorkplaceAction, WorkplaceObservation
    from server.workplace_environment import WorkplaceEnvironment


app = create_app(
    WorkplaceEnvironment,
    WorkplaceAction,
    WorkplaceObservation,
    env_name="workplace_policy",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the OpenEnv HTTP server."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the workplace policy environment.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # The literal main() marker keeps older OpenEnv validators happy.
    main(host=args.host, port=args.port)
