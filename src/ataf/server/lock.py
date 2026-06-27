"""Global build lock and long-poll waiter queue.

Only one tool is built at a time across the entire ATAF server. This
module implements that lock and the long-poll machinery that lets
other agents wait without busy-polling.

Why a global lock instead of per-tool (see DESIGN.md §5):
  Per-tool locking requires recognizing "the same tool" across two
  submissions, which devolves into fragile name normalization or
  unreliable semantic similarity. A global lock sidesteps the question.
  Builds are infrequent and fast.

The flow:

    Agent A: POST /tools/propose
        -> server calls BuildLock.acquire() -> returns Token immediately
        -> server builds the tool (~100ms)
        -> server calls token.release() -> notifies waiters

    Agent B: POST /tools/propose  (while A's build is in flight)
        -> server calls BuildLock.acquire(no_wait=True) -> returns None
        -> server returns 202 WAIT with a poll_url
        -> agent B polls GET /tools/propose/wait/{poll_url}
        -> handler calls BuildLock.wait_for_release()
        -> wakes when A finishes
        -> returns the refreshed catalog

    Agent C: GET /tools  (while A's build is in flight)
        -> handler calls BuildLock.wait_for_release()
        -> wakes when A finishes
        -> returns the refreshed catalog

Implementation: a ``threading.Event``. Set means "no build in flight,
proceed." Cleared means "a build is in flight, wait." The token
returned by ``acquire()`` is the only way to flip the event back to
set, so the lock is always released even if the build raises.
"""

from __future__ import annotations

import secrets
from threading import Event, Lock
from typing import Optional


class BuildToken:
    """Opaque handle returned by ``BuildLock.acquire()``.

    The token's only purpose is to release the lock. The caller MUST
    release it (use a ``try/finally`` or a context manager) — otherwise
    every subsequent ``acquire()`` will block forever.
    """

    def __init__(self, lock: "BuildLock", token_id: str) -> None:
        self._lock = lock
        self.id = token_id
        self._released = False

    def release(self) -> None:
        """Release the build lock and wake all waiters. Idempotent."""

        # Idempotency lets ``__exit__`` call us even if the user already
        # released manually inside the with-block, without raising.
        if self._released:
            return
        self._released = True
        self._lock._release(self)

    # Context manager support for the common ``with lock.acquire() as token``
    # pattern. The token IS the context — release is automatic on exit.
    def __enter__(self) -> "BuildToken":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class BuildLock:
    """Process-wide single-build lock with long-poll waiter support.

    Usage:

        lock = BuildLock()
        token = lock.acquire(no_wait=True)
        if token is None:
            # Another build is in flight; long-poll the wait endpoint.
            poll_url = lock.new_poll_token()
            return ProposeWaitResponse(poll_url=f"/tools/propose/wait/{poll_url}", eta_seconds=5)
        try:
            # ... do the build ...
        finally:
            token.release()

    The ``no_wait=True`` flag exists because in our protocol we never
    want the propose endpoint to silently block — we want it to return
    202 WAIT immediately and let the agent poll. Other callers (e.g.,
    ``GET /tools``) can block via ``wait_for_release()`` directly.
    """

    def __init__(self) -> None:
        # _is_idle is "set" when no build is in flight (the normal state).
        # We use Event because it gives us cheap, multi-waiter signaling.
        self._is_idle = Event()
        self._is_idle.set()

        # Guards the transition from "set" to "cleared" so two threads
        # can't both grab the lock simultaneously. Event itself is
        # thread-safe for individual ops but not for the
        # check-then-clear race we're trying to avoid.
        self._acquire_guard = Lock()

        # The currently held token, if any. Used to validate releases
        # — only the token-holder can release, not random callers.
        self._current_token: Optional[BuildToken] = None

    def acquire(self, *, no_wait: bool = False, timeout: float | None = None) -> Optional[BuildToken]:
        """Try to acquire the build lock.

        Args:
            no_wait: If True, return None immediately when the lock is
                held instead of blocking. Used by the propose endpoint
                so we can return 202 WAIT to the caller.
            timeout: If no_wait is False, the max seconds to wait.
                ``None`` means wait forever.

        Returns:
            A ``BuildToken`` on success, or None if the lock could not
            be acquired (no_wait=True and lock held, or timeout fired).
        """

        # Fast path: no_wait means we attempt one non-blocking acquire.
        if no_wait:
            with self._acquire_guard:
                if self._is_idle.is_set():
                    self._is_idle.clear()
                    token = BuildToken(self, secrets.token_hex(8))
                    self._current_token = token
                    return token
                return None

        # Blocking path: wait for the lock to become free, then try to
        # claim it under the guard. There can be a tiny race where two
        # waiters both see "idle" and one wins; the loser loops and
        # waits again.
        while True:
            # Wait for the current build (if any) to finish.
            if not self._is_idle.wait(timeout=timeout):
                # Timed out.
                return None
            with self._acquire_guard:
                if self._is_idle.is_set():
                    self._is_idle.clear()
                    token = BuildToken(self, secrets.token_hex(8))
                    self._current_token = token
                    return token
                # Lost the race; loop and wait again.
                continue

    def wait_for_release(self, timeout: float | None = None) -> bool:
        """Block until the current build (if any) finishes.

        Used by handlers that want to long-poll without acquiring the
        lock themselves — for example, ``GET /tools`` and the propose
        wait endpoint.

        Args:
            timeout: Max seconds to wait. None means wait forever.

        Returns:
            True if a build finished (or no build was in flight),
            False if the timeout fired first.
        """

        # If the lock is already free, this returns immediately.
        return self._is_idle.wait(timeout=timeout)

    def is_building(self) -> bool:
        """Return True if a build is currently in flight."""

        # Event is "set" when idle, so "not set" means a build is running.
        return not self._is_idle.is_set()

    # ------------------------------------------------------------------
    # Internal: called by BuildToken.release()
    # ------------------------------------------------------------------

    def _release(self, token: BuildToken) -> None:
        """Release the lock. Called by ``BuildToken.release()``.

        Args:
            token: The token that holds the lock. Anything else raises.
        """

        with self._acquire_guard:
            # Defensive check: only the current token-holder may release.
            # This catches bugs where a stale token gets released after
            # someone else has acquired.
            if self._current_token is None or self._current_token.id != token.id:
                # Idempotent: silently ignore. Logging would be a nice
                # add but we deliberately keep this module log-free so
                # tests don't need to mock out the event log.
                return
            self._current_token = None
            self._is_idle.set()
