"""Standalone subprocess entry point for executing exactly one tool call.

This file is run in a **fresh Python interpreter** by ``executor.py`` —
it is never imported by the server. Running tool code in a separate
process is what gives us a hard timeout (the parent can kill it) and a
blast radius that can't corrupt server state.

Protocol (all over stdin/stdout, one JSON object each way):

    stdin  : {"code": "<source>", "func_name": "circle_area", "args": {...}}
    stdout : {"ok": true,  "result": <json value>}
             {"ok": false, "error_type": "ZeroDivisionError", "error": "..."}

The runner is intentionally dependency-free (stdlib only) so it starts
fast and has nothing of ours in scope that a tool could reach.

v0.1 trust model (DESIGN.md §9): the LLM is assumed honest. There is NO
sandboxing here yet — the tool runs with the same privileges as the
server process. Real isolation (WASM / microVM) is v0.3 work.
"""

import json
import sys


def main() -> None:
    """Read one job from stdin, run it, and print one JSON result line."""

    # Read the entire stdin payload. The parent writes it then closes the
    # pipe, so a single read() returns the whole JSON document.
    raw = sys.stdin.read()
    payload = json.loads(raw)

    code = payload["code"]
    func_name = payload["func_name"]
    args = payload["args"]

    try:
        # Execute the tool's source in a fresh namespace, then pull out
        # the function the registry recorded by name and call it with the
        # supplied keyword arguments.
        namespace: dict = {}
        exec(compile(code, "<tool>", "exec"), namespace)
        func = namespace[func_name]
        result = func(**args)

        # The result must be JSON-serializable to cross back to the
        # parent. Probe it here so a non-serializable return becomes a
        # clean error instead of a stdout decode failure upstream.
        json.dumps(result)

        # Success — emit the result as the final stdout line.
        print(json.dumps({"ok": True, "result": result}))
    except Exception as exc:  # noqa: BLE001 - we want to report ANY failure
        # Any exception (in the code, the call, or serialization) is
        # reported structurally so the parent can surface a clean message.
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        )


if __name__ == "__main__":
    main()
