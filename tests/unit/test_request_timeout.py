# -*- coding: utf-8 -*-
"""Test request timeout (TODO-006)."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_timeout_config_exists():
    """Test that AGENT_TIMEOUT configuration exists."""
    try:
        from config.langvel import config

        assert hasattr(config, 'AGENT_TIMEOUT')
        assert isinstance(config.AGENT_TIMEOUT, int)
        assert config.AGENT_TIMEOUT > 0

        print(f"[PASS] AGENT_TIMEOUT config exists: {config.AGENT_TIMEOUT}s")
        return True
    except Exception as e:
        print(f"[FAIL] Config error: {e}")
        return False


def test_asyncio_imported():
    """Test that asyncio is imported in server."""
    try:
        from langvel import server
        import inspect

        source = inspect.getsource(server)
        assert "import asyncio" in source

        print("[PASS] asyncio imported in server")
        return True
    except Exception as e:
        print(f"[FAIL] Import check error: {e}")
        return False


def test_invoke_agent_has_timeout():
    """Test that invoke_agent uses asyncio.wait_for with timeout."""
    from langvel.server import invoke_agent
    import inspect

    source = inspect.getsource(invoke_agent)

    checks = [
        "asyncio.wait_for" in source,
        "timeout=config.AGENT_TIMEOUT" in source,
        "asyncio.TimeoutError" in source,
        "504" in source,  # Gateway Timeout status code
    ]

    if not all(checks):
        print("[FAIL] Timeout implementation not found in invoke_agent")
        print(f"  asyncio.wait_for: {'✓' if checks[0] else '✗'}")
        print(f"  timeout config: {'✓' if checks[1] else '✗'}")
        print(f"  TimeoutError handler: {'✓' if checks[2] else '✗'}")
        print(f"  504 status: {'✓' if checks[3] else '✗'}")
        return False

    print("[PASS] invoke_agent has timeout implementation")
    return True


def test_timeout_returns_504():
    """Test that timeout returns 504 Gateway Timeout."""
    from langvel.server import invoke_agent
    import inspect

    source = inspect.getsource(invoke_agent)

    # Check for 504 status code in TimeoutError handler
    if "504" not in source or "TimeoutError" not in source:
        print("[FAIL] 504 status code not found in timeout handler")
        return False

    print("[PASS] Timeout returns 504 Gateway Timeout")
    return True


def test_timeout_logging():
    """Test that timeout events are logged."""
    from langvel.server import invoke_agent
    import inspect

    source = inspect.getsource(invoke_agent)

    # Check for logging in timeout handler
    timeout_section = source.split("TimeoutError")[1].split("except")[0] if "TimeoutError" in source else ""

    if "logger.error" not in timeout_section:
        print("[FAIL] Timeout logging not found")
        return False

    if "agent_path" not in timeout_section or "timeout" not in timeout_section:
        print("[FAIL] Timeout logging missing context")
        return False

    print("[PASS] Timeout events are logged with context")
    return True


def test_timeout_error_message():
    """Test that timeout error message is informative."""
    from langvel.server import invoke_agent
    import inspect

    source = inspect.getsource(invoke_agent)

    # Check for informative error message
    if "Agent execution timeout" not in source:
        print("[FAIL] Timeout error message not informative")
        return False

    if "seconds" not in source:
        print("[FAIL] Timeout duration not included in message")
        return False

    print("[PASS] Timeout error message is informative")
    return True


def test_timeout_default_value():
    """Test that default timeout is reasonable (5 minutes)."""
    from config.langvel import config

    # Default should be 300 seconds (5 minutes)
    assert config.AGENT_TIMEOUT == 300, f"Expected 300s, got {config.AGENT_TIMEOUT}s"

    print(f"[PASS] Default timeout is {config.AGENT_TIMEOUT}s (5 minutes)")
    return True


def test_httpexception_reraised():
    """Test that HTTPException is re-raised correctly."""
    from langvel.server import invoke_agent
    import inspect

    source = inspect.getsource(invoke_agent)

    # Check for HTTPException handling
    if "except HTTPException:" not in source and "except HTTPException" not in source:
        print("[FAIL] HTTPException not handled separately")
        return False

    # Should have a re-raise
    exception_handlers = source.split("except")
    for handler in exception_handlers:
        if "HTTPException" in handler:
            if "raise" in handler.split("except")[0] if "except" in handler else handler:
                print("[PASS] HTTPException is re-raised correctly")
                return True

    print("[FAIL] HTTPException not re-raised")
    return False


def test_timeout_config_env_var():
    """Test that AGENT_TIMEOUT can be configured via environment variable."""
    import os
    from importlib import reload
    from config import langvel

    # Save original value
    original_value = os.environ.get('AGENT_TIMEOUT')

    try:
        # Set custom timeout
        os.environ['AGENT_TIMEOUT'] = '600'

        # Reload config to pick up new value
        reload(langvel)

        assert langvel.config.AGENT_TIMEOUT == 600

        print("[PASS] AGENT_TIMEOUT configurable via environment variable")
        return True
    except Exception as e:
        print(f"[FAIL] Environment variable configuration error: {e}")
        return False
    finally:
        # Restore original value
        if original_value:
            os.environ['AGENT_TIMEOUT'] = original_value
        else:
            os.environ.pop('AGENT_TIMEOUT', None)

        # Reload to restore original config
        reload(langvel)


if __name__ == '__main__':
    print("="*60)
    print("Testing Request Timeout (TODO-006)")
    print("="*60)
    print()

    tests = [
        ("AGENT_TIMEOUT config exists", test_timeout_config_exists),
        ("asyncio imported", test_asyncio_imported),
        ("invoke_agent has timeout", test_invoke_agent_has_timeout),
        ("Timeout returns 504", test_timeout_returns_504),
        ("Timeout logging", test_timeout_logging),
        ("Timeout error message", test_timeout_error_message),
        ("Default timeout value", test_timeout_default_value),
        ("HTTPException re-raised", test_httpexception_reraised),
        ("Config via env var", test_timeout_config_env_var),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"Running: {name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*60)

    sys.exit(0 if failed == 0 else 1)
