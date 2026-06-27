"""Tests for the global BuildLock + long-poll waiter mechanics."""

import threading
import time

from ataf.server.lock import BuildLock


def test_acquire_no_wait_succeeds_when_idle() -> None:
    """A fresh lock is idle, so a no_wait acquire must return a token."""

    lock = BuildLock()
    token = lock.acquire(no_wait=True)
    assert token is not None
    assert not lock.is_building() is False  # double-negate for clarity
    assert lock.is_building() is True

    token.release()
    assert lock.is_building() is False


def test_acquire_no_wait_returns_none_when_held() -> None:
    """While the lock is held, a second no_wait acquire must return None."""

    lock = BuildLock()
    first = lock.acquire(no_wait=True)
    assert first is not None

    second = lock.acquire(no_wait=True)
    assert second is None

    first.release()


def test_release_wakes_blocking_waiters() -> None:
    """A thread blocked on wait_for_release() must wake when the lock is released."""

    lock = BuildLock()
    holder = lock.acquire(no_wait=True)
    assert holder is not None

    wake_time: list[float | None] = [None]

    def waiter() -> None:
        # Block until the holder releases. Record when we wake.
        lock.wait_for_release()
        wake_time[0] = time.monotonic()

    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()

    # Give the waiter time to actually block on the event
    time.sleep(0.05)
    release_time = time.monotonic()
    holder.release()

    waiter_thread.join(timeout=1.0)
    assert wake_time[0] is not None
    # The waiter should wake very soon after release (well under 100ms)
    assert wake_time[0] - release_time < 0.1


def test_wait_for_release_times_out() -> None:
    """wait_for_release with a tight timeout must return False if no release."""

    lock = BuildLock()
    held = lock.acquire(no_wait=True)
    assert held is not None

    # Don't release. wait_for_release should give up.
    started = time.monotonic()
    result = lock.wait_for_release(timeout=0.05)
    elapsed = time.monotonic() - started

    assert result is False
    assert 0.04 < elapsed < 0.5  # wide bounds for CI jitter
    held.release()


def test_context_manager_releases_on_exit() -> None:
    """Using BuildToken as a context manager must release on exit."""

    lock = BuildLock()
    token = lock.acquire(no_wait=True)
    assert token is not None

    with token:
        assert lock.is_building() is True

    assert lock.is_building() is False


def test_release_is_idempotent() -> None:
    """Calling release() twice on the same token must not raise or break the lock."""

    lock = BuildLock()
    token = lock.acquire(no_wait=True)
    assert token is not None
    token.release()
    token.release()  # second call: no-op

    # The lock should still be acquirable
    next_token = lock.acquire(no_wait=True)
    assert next_token is not None
    next_token.release()


def test_stale_token_release_is_ignored() -> None:
    """If someone releases a stale token after someone else has acquired,
    the new holder must keep their lock."""

    lock = BuildLock()
    first = lock.acquire(no_wait=True)
    assert first is not None
    first.release()

    second = lock.acquire(no_wait=True)
    assert second is not None

    # Stale release of an already-released token
    first.release()

    # The second holder must still own the lock
    assert lock.is_building() is True
    second.release()
