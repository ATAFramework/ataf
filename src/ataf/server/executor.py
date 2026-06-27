"""Tool executor — runs a deployed tool's code in a subprocess.

The executor is the only component that actually *runs* tool code. It
loads the source from the registry (which integrity-checks it on read),
hands it to a fresh Python interpreter via ``executor_runner.py``, and
marshals the result back.

Why a subprocess (DESIGN.md §8.3):
  * **Timeout.** A runaway tool (infinite loop) can be killed without
    taking the server down.
  * **Isolation of state.** The tool can't see or mutate server objects;
    it starts in a clean interpreter.

What the executor does NOT do in v0.1:
  * It does not sandbox. The subprocess has the same OS privileges as the
    server. The v0.1 trust model assumes an honest LLM; real sandboxing
    (WASM / microVM) is deferred to v0.3.
  * It does not check governance. Whether a tool is allowed to run at all
    is ``governance.py``'s job; by the time code reaches the executor the
    caller has already confirmed the tool is invokable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .registry import Registry


# The standalone child-process entry point. Resolved once at import time
# relative to this file so it works regardless of the process CWD.
_RUNNER_PATH = Path(__file__).parent / "executor_runner.py"


class ToolExecutionError(Exception):
    """Raised when a tool fails to execute or returns an error.

    Covers three distinct failure modes, distinguished by ``code``:

      * ``TOOL_TIMEOUT``         — the tool exceeded its time budget.
      * ``TOOL_EXECUTION_ERROR`` — the tool raised, or returned a value
        that isn't JSON-serializable.
      * ``TOOL_RUNNER_ERROR``    — the subprocess itself failed (crashed
        before producing a result line). Should be rare.

    The caller maps this to a ``500`` response carrying ``code`` and
    ``message`` in the standard error envelope.

    Attributes:
        code: Stable machine-readable error code (see above).
        message: Human-readable explanation, safe to surface to the LLM
            as a tool-result.
    """

    def __init__(self, message: str, *, code: str = "TOOL_EXECUTION_ERROR") -> None:
        """Build the error.

        Args:
            message: Human-readable explanation.
            code: Stable machine-readable code; defaults to
                ``TOOL_EXECUTION_ERROR``.
        """

        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class Executor:
    """Runs deployed tools in short-lived subprocesses.

    One instance per ATAF server, sharing the registry with the rest of
    the server. Stateless beyond its configured timeout, so it is safe to
    call concurrently from many request threads.
    """

    def __init__(self, registry: Registry, timeout_seconds: float = 5.0) -> None:
        """Construct the executor.

        Args:
            registry: The shared tool registry. Used to load (and
                integrity-check) the tool's source and to record a
                successful invocation.
            timeout_seconds: Wall-clock budget for a single tool call.
                Tools exceeding this are killed and reported as
                ``TOOL_TIMEOUT``.
        """

        self._registry = registry
        self._timeout = timeout_seconds

    def invoke(self, tool_id: str, args: dict) -> object:
        """Execute one tool call and return its result.

        Args:
            tool_id: The tool to run. Must already exist and be invokable
                (governance is checked by the caller, not here).
            args: Keyword arguments to pass to the tool's function.

        Returns:
            Whatever the tool returned, as a JSON-compatible Python value.

        Raises:
            KeyError: If the tool_id is not in the registry.
            ToolExecutionError: On timeout, a tool-raised exception, a
                non-serializable return value, or a runner crash.
        """

        # Load the source (raises KeyError if unknown; IntegrityError if
        # the on-disk file was tampered with) and the row for the function
        # name the registry recorded at deploy time.
        code = self._registry.get_code(tool_id)
        row = self._registry.get(tool_id)
        if row is None:
            # get_code already raises on a missing tool, so this is just a
            # belt-and-suspenders guard for the type-checker and races.
            raise KeyError(f"unknown tool_id: {tool_id!r}")

        # Build the job payload for the runner: the source, which function
        # to call, and the arguments to call it with.
        payload = json.dumps(
            {"code": code, "func_name": row.name, "args": args}
        )

        # Run the tool in a fresh interpreter with a hard timeout. We
        # capture both streams as text so we can parse stdout and surface
        # stderr on failure.
        try:
            completed = subprocess.run(
                [sys.executable, str(_RUNNER_PATH)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            # The child was killed for exceeding its budget.
            raise ToolExecutionError(
                f"tool {tool_id!r} timed out after {self._timeout}s",
                code="TOOL_TIMEOUT",
            )

        # The runner prints exactly one JSON result line on success or a
        # handled error. No usable stdout means the subprocess crashed
        # before reporting — surface stderr to aid debugging.
        stdout = completed.stdout.strip()
        if not stdout:
            raise ToolExecutionError(
                f"tool {tool_id!r} produced no output "
                f"(exit {completed.returncode}): {completed.stderr.strip()}",
                code="TOOL_RUNNER_ERROR",
            )

        # Parse the final stdout line as the result envelope. Taking the
        # last line is defensive in case the tool itself printed to stdout.
        last_line = stdout.splitlines()[-1]
        outcome = json.loads(last_line)

        # A tool-level failure (raised exception, non-serializable return)
        # becomes a TOOL_EXECUTION_ERROR with the child's reported reason.
        if not outcome.get("ok"):
            error_type = outcome.get("error_type", "Error")
            error_message = outcome.get("error", "unknown error")
            raise ToolExecutionError(f"{error_type}: {error_message}")

        # Success — record the invocation (bumps call_count / last_called)
        # and hand the result back to the caller.
        self._registry.record_invocation(tool_id)
        return outcome["result"]
