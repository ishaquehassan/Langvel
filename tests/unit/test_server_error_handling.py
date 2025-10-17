# -*- coding: utf-8 -*-
"""Test server error handling (TODO-002)."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_error_handler_imports():
    """Test that error handling imports are present."""
    try:
        from langvel.server import app, global_exception_handler, logger
        import uuid
        import logging

        # Verify imports work
        assert app is not None
        assert global_exception_handler is not None
        assert logger is not None

        print("[PASS] Error handler imports verified")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_debug_mode_shows_details():
    """Test that DEBUG mode shows detailed errors."""
    from langvel.server import global_exception_handler
    from fastapi import Request
    from config.langvel import config

    # Mock request
    class MockClient:
        host = "127.0.0.1"

    class MockRequest:
        method = "POST"
        class url:
            path = "/test"
        client = MockClient()

    request = MockRequest()

    # Test exception
    test_exception = ValueError("Test error message")

    # Save original DEBUG setting
    old_debug = config.DEBUG

    try:
        # Set DEBUG mode
        config.DEBUG = True

        # Call handler
        import asyncio
        response = asyncio.run(global_exception_handler(request, test_exception))

        # Verify response contains detailed error
        content = response.body.decode('utf-8')
        assert "Test error message" in content or "ValueError" in content
        assert "trace_id" in content

        print("[PASS] DEBUG mode shows detailed errors")
        return True
    except Exception as e:
        print(f"[FAIL] DEBUG mode test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        config.DEBUG = old_debug


def test_production_mode_hides_details():
    """Test that production mode hides error details."""
    from langvel.server import global_exception_handler
    from fastapi import Request
    from config.langvel import config

    # Mock request
    class MockClient:
        host = "127.0.0.1"

    class MockRequest:
        method = "POST"
        class url:
            path = "/test"
        client = MockClient()

    request = MockRequest()

    # Test exception with sensitive info
    test_exception = ValueError("Database password: secret123")

    # Save original DEBUG setting
    old_debug = config.DEBUG

    try:
        # Set production mode
        config.DEBUG = False

        # Call handler
        import asyncio
        response = asyncio.run(global_exception_handler(request, test_exception))

        # Verify response does NOT contain sensitive error
        content = response.body.decode('utf-8')
        assert "secret123" not in content
        assert "Database password" not in content
        assert "Internal server error" in content
        assert "trace_id" in content

        print("[PASS] Production mode hides error details")
        return True
    except Exception as e:
        print(f"[FAIL] Production mode test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        config.DEBUG = old_debug


def test_trace_id_generated():
    """Test that trace IDs are generated."""
    from langvel.server import global_exception_handler
    from fastapi import Request
    import json

    # Mock request
    class MockClient:
        host = "127.0.0.1"

    class MockRequest:
        method = "GET"
        class url:
            path = "/test"
        client = MockClient()

    request = MockRequest()
    test_exception = RuntimeError("Test error")

    try:
        # Call handler twice
        import asyncio
        response1 = asyncio.run(global_exception_handler(request, test_exception))
        response2 = asyncio.run(global_exception_handler(request, test_exception))

        # Parse responses
        content1 = json.loads(response1.body.decode('utf-8'))
        content2 = json.loads(response2.body.decode('utf-8'))

        # Verify trace IDs exist and are different
        assert 'trace_id' in content1
        assert 'trace_id' in content2
        assert content1['trace_id'] != content2['trace_id']

        # Verify trace ID format (UUID)
        assert len(content1['trace_id']) == 36  # UUID length

        print("[PASS] Trace IDs generated correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Trace ID test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*60)
    print("Testing Server Error Handling (TODO-002)")
    print("="*60)
    print()

    tests = [
        ("Error handler imports", test_error_handler_imports),
        ("DEBUG mode shows details", test_debug_mode_shows_details),
        ("Production mode hides details", test_production_mode_hides_details),
        ("Trace ID generation", test_trace_id_generated),
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
