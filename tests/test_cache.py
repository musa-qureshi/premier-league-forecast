"""Unit tests for app/backend/cache.py.

background_refresh_loop() itself (an infinite sleep-then-refresh loop) is
deliberately not unit-tested - there's nothing meaningful to assert about
an infinite loop without either waiting hours or mocking asyncio.sleep
into meaninglessness. What IS tested is refresh_once(): the one-shot
refresh attempt with error handling, which is the part that actually
matters to get right - a bad refresh must never crash the loop and
silently stop all future automatic refreshes for the rest of the
process's life.
"""

import app.backend.cache as cache_module


def test_refresh_once_returns_true_on_success(monkeypatch):
    monkeypatch.setattr(cache_module, "get_forecast", lambda force_refresh: "a forecast")
    assert cache_module.refresh_once() is True

    status = cache_module.cache_status()
    assert status["last_background_refresh_error"] is None
    assert status["last_background_refresh_attempt_at"] is not None


def test_refresh_once_swallows_exceptions_and_returns_false(monkeypatch):
    def raise_network_error(force_refresh):
        raise ConnectionError("could not reach github.com")

    monkeypatch.setattr(cache_module, "get_forecast", raise_network_error)

    # The key property: this must NOT raise. A failed background refresh
    # attempt should be recorded, not crash the loop it runs inside.
    result = cache_module.refresh_once()
    assert result is False

    status = cache_module.cache_status()
    assert "could not reach github.com" in status["last_background_refresh_error"]
    assert status["last_background_refresh_attempt_at"] is not None


def test_refresh_once_success_clears_a_previous_error(monkeypatch):
    monkeypatch.setattr(
        cache_module, "get_forecast",
        lambda force_refresh: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    cache_module.refresh_once()
    assert cache_module.cache_status()["last_background_refresh_error"] is not None

    monkeypatch.setattr(cache_module, "get_forecast", lambda force_refresh: "ok now")
    cache_module.refresh_once()
    assert cache_module.cache_status()["last_background_refresh_error"] is None
