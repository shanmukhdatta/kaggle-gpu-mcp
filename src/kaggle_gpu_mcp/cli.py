"""Portable CLI for stdio MCP, worker and local acceptance preparation."""
import argparse
import json
import shutil
import sys

from .config import Settings
from .demo import install_demo
from .service import Service
from .worker import Worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle GPU MCP")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("serve", help="Run MCP via stdio (worker must run separately)")
    worker = commands.add_parser("worker", help="Run background worker")
    worker.add_argument("--once", action="store_true", help="Perform one scheduler tick")
    commands.add_parser("demo", help="Create editable example scripts in workspace")
    commands.add_parser("check", help="Check Kaggle connection")
    commands.add_parser("doctor", help="Run a credential-safe local readiness check")
    commands.add_parser("jobs", help="List recent jobs")
    details = commands.add_parser("job", help="Inspect one job")
    details.add_argument("id")
    run = commands.add_parser("run-demo", help="Queue CPU prep and GPU training example")
    run.add_argument("--approve", action="store_true", help="Explicitly authorize two remote jobs")
    args = parser.parse_args()
    settings = Settings.from_env()
    try:
        if args.action == "serve":
            from .server import create_server
            create_server(settings).run(transport="stdio")
        elif args.action == "worker":
            w = Worker(settings)
            w.tick() if args.once else w.run_forever()
        elif args.action == "demo":
            print(json.dumps(install_demo(settings), indent=2))
        elif args.action == "check":
            print(json.dumps(Service(settings).provider.check(), indent=2))
        elif args.action == "doctor":
            # Do not print usernames, tokens, passwords or dataset file contents.
            from importlib.util import find_spec
            report = {
                "workspace_exists": settings.workspace.is_dir(),
                "kaggle_cli_on_path": bool(shutil.which(settings.kaggle_bin)),
                "mcp_sdk_installed": find_spec("mcp") is not None,
                "username_configured": bool(settings.username),
                "submissions_explicitly_enabled": settings.allow_submissions,
                "smtp_notification_configured": bool(settings.email_to and settings.smtp_host
                                                     and settings.smtp_sender),
                "recommendation": "Run 'check' after configuring Kaggle credentials",
            }
            print(json.dumps(report, indent=2))
        elif args.action == "jobs":
            print(json.dumps(Service(settings).db.list(), indent=2))
        elif args.action == "job":
            print(json.dumps(Service(settings).details(args.id), indent=2))
        elif args.action == "run-demo":
            if not args.approve:
                parser.error("Use --approve to explicitly authorize Kaggle execution")
            service = Service(settings)
            print(json.dumps(service.workflow("demo/prep.py", "demo/train.py", approved=True), indent=2))
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
