# -*- coding: utf-8 -*-
"""Test request size limit middleware (TODO-003)."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_middleware_imports():
    """Test that middleware imports correctly."""
    try:
        from langvel.server import RequestSizeLimitMiddleware, app
        from starlette.middleware.base import BaseHTTPMiddleware

        # Verify middleware class exists
        assert RequestSizeLimitMiddleware is not None
        assert issubclass(RequestSizeLimitMiddleware, BaseHTTPMiddleware)

        # Verify middleware is registered (check app middleware stack)
        middleware_found = False
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == 'RequestSizeLimitMiddleware':
                middleware_found = True
                break

        assert middleware_found, "RequestSizeLimitMiddleware not found in app middleware stack"

        print("[PASS] Middleware imports and registration verified")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_small_request_allowed():
    """Test that small requests are allowed through."""
    from fastapi.testclient import TestClient
    from langvel.server import app

    client = TestClient(app)

    # Small request should pass (root endpoint)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Langvel Server"

    print("[PASS] Small requests allowed")
    return True


def test_large_request_blocked():
    """Test that oversized requests are blocked."""
    from langvel.server import RequestSizeLimitMiddleware
    from fastapi import Request, HTTPException as FastHTTPException
    from unittest.mock import Mock, AsyncMock
    import asyncio

    try:
        # Create middleware
        middleware = RequestSizeLimitMiddleware(app=None, max_size=10_000_000)

        # Mock request with large content-length
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = Mock()
        mock_request.url.path = "/test"
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"content-length": "20000000"}  # 20MB

        # Mock call_next
        mock_call_next = AsyncMock()

        # Should raise HTTPException
        async def test_dispatch():
            try:
                await middleware.dispatch(mock_request, mock_call_next)
                return False  # Should not reach here
            except FastHTTPException as e:
                return e.status_code == 413

        result = asyncio.run(test_dispatch())
        assert result, "Expected 413 HTTPException"

        print("[PASS] Large requests blocked with 413")
        return True
    except Exception as e:
        print(f"[FAIL] Large request test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_content_length():
    """Test that invalid Content-Length headers are rejected."""
    from langvel.server import RequestSizeLimitMiddleware
    from fastapi import Request, HTTPException as FastHTTPException
    from unittest.mock import Mock, AsyncMock
    import asyncio

    try:
        # Create middleware
        middleware = RequestSizeLimitMiddleware(app=None)

        # Mock request with invalid content-length
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = Mock()
        mock_request.url.path = "/test"
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"content-length": "not-a-number"}

        # Mock call_next
        mock_call_next = AsyncMock()

        # Should raise HTTPException with 400
        async def test_dispatch():
            try:
                await middleware.dispatch(mock_request, mock_call_next)
                return False  # Should not reach here
            except FastHTTPException as e:
                return e.status_code == 400

        result = asyncio.run(test_dispatch())
        assert result, "Expected 400 HTTPException"

        print("[PASS] Invalid Content-Length rejected with 400")
        return True
    except Exception as e:
        print(f"[FAIL] Invalid Content-Length test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_requests_not_checked():
    """Test that GET requests bypass size check."""
    from fastapi.testclient import TestClient
    from langvel.server import app

    client = TestClient(app)

    try:
        # GET requests should not be checked (only POST/PUT/PATCH)
        # Even with large Content-Length, should work for GET
        response = client.get(
            "/health",
            headers={"Content-Length": "20000000"}  # 20MB
        )

        # Should succeed because GET is not checked
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        print("[PASS] GET requests bypass size check")
        return True
    except Exception as e:
        print(f"[FAIL] GET request test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_middleware_default_size():
    """Test that middleware has correct default size limit."""
    from langvel.server import RequestSizeLimitMiddleware

    try:
        # Create middleware with default
        middleware = RequestSizeLimitMiddleware(app=None)

        # Verify default is 10MB
        assert middleware.max_size == 10_000_000, f"Expected 10000000, got {middleware.max_size}"

        print("[PASS] Middleware default size is 10MB")
        return True
    except Exception as e:
        print(f"[FAIL] Default size test: {e}")
        return False


if __name__ == '__main__':
    print("="*60)
    print("Testing Request Size Limit Middleware (TODO-003)")
    print("="*60)
    print()

    tests = [
        ("Middleware imports and registration", test_middleware_imports),
        ("Small requests allowed", test_small_request_allowed),
        ("Large requests blocked", test_large_request_blocked),
        ("Invalid Content-Length rejected", test_invalid_content_length),
        ("GET requests bypass size check", test_get_requests_not_checked),
        ("Middleware default size", test_middleware_default_size),
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
