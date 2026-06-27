"""``ataf-admin`` — command-line tool for human review of deployed tools.

New tools deploy as ``PENDING_REVIEW`` and cannot be invoked until a
human approves them (DESIGN.md §6). This CLI is that human's interface in
v0.1 (a web UI lands in v0.2).

It talks **directly to the registry on disk**, not over HTTP — the admin
is assumed to be on the same machine as the server's data directory. This
keeps the v0.1 admin path dependency-free and usable even when the server
process is down.

Usage:

    ataf-admin list                       # show every tool + status
    ataf-admin approve circle_area_v1     # mark AUTHORIZED
    ataf-admin reject   circle_area_v1    # mark UNAUTHORIZED

    # point at a non-default data dir:
    ataf-admin --data-dir /srv/ataf_data list
"""

from __future__ import annotations

import argparse
import sys

from .eventlog import DeploymentEventLog
from .registry import Registry
from .storage import default_storage


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ataf-admin`` console script.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``). Exposed for
            tests so they can drive the CLI without touching ``sys.argv``.

    Returns:
        Process exit code: 0 on success, non-zero on error.
    """

    # --- Argument parsing ---
    parser = argparse.ArgumentParser(
        prog="ataf-admin",
        description="Review and govern ATAF-deployed tools.",
    )
    parser.add_argument(
        "--data-dir",
        default="ataf_data",
        help="ATAF data directory (default: ./ataf_data)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Three subcommands: list, approve <id>, reject <id>.
    subparsers.add_parser("list", help="List all tools and their status.")
    approve_parser = subparsers.add_parser(
        "approve", help="Authorize a tool so it can be invoked."
    )
    approve_parser.add_argument("tool_id")
    reject_parser = subparsers.add_parser(
        "reject", help="Reject a tool so it can never be invoked."
    )
    reject_parser.add_argument("tool_id")

    args = parser.parse_args(argv)

    # --- Open the registry over the chosen data dir ---
    paths = default_storage(args.data_dir)
    paths.ensure_exists()
    registry = Registry(paths)
    registry.initialize()
    event_log = DeploymentEventLog(paths.deployment_log)

    # --- Dispatch ---
    try:
        if args.command == "list":
            return _cmd_list(registry)
        if args.command == "approve":
            return _cmd_set_status(
                registry, event_log, args.tool_id, "AUTHORIZED", "approve"
            )
        if args.command == "reject":
            return _cmd_set_status(
                registry, event_log, args.tool_id, "UNAUTHORIZED", "reject"
            )
    finally:
        registry.close()

    # argparse with required=True makes this unreachable, but keeps the
    # function total for the type-checker.
    return 2


def _cmd_list(registry: Registry) -> int:
    """Print every tool with its status, call count, and creation time."""

    tools = registry.list_all()
    if not tools:
        print("No tools deployed yet.")
        return 0

    # Fixed-width columns for a readable terminal table.
    print(f"{'TOOL_ID':<24} {'STATUS':<16} {'CALLS':>6}  CREATED")
    print("-" * 72)
    for row in tools:
        print(
            f"{row.tool_id:<24} {row.status:<16} {row.call_count:>6}  "
            f"{row.created_at}"
        )
    return 0


def _cmd_set_status(
    registry: Registry,
    event_log: DeploymentEventLog,
    tool_id: str,
    new_status: str,
    event_name: str,
) -> int:
    """Flip a tool's status and log the action, or report an unknown id.

    Args:
        registry: The open registry.
        event_log: The deployment event log (records the admin action).
        tool_id: The tool to mutate.
        new_status: ``AUTHORIZED`` or ``UNAUTHORIZED``.
        event_name: ``approve`` or ``reject`` (for the audit log).

    Returns:
        0 on success, 1 if the tool_id doesn't exist.
    """

    try:
        registry.set_status(tool_id, new_status)
    except KeyError:
        print(f"error: no tool with id {tool_id!r}", file=sys.stderr)
        return 1

    # Record the human action in the audit trail, then confirm to stdout.
    event_log.record(event_name, tool_id=tool_id, actor="cli")
    print(f"{tool_id} -> {new_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
