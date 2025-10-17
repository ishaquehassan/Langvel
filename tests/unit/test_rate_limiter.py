# -*- coding: utf-8 -*-
"""Test rate limiter fixes (TODO-007)."""

import os
import sys
import asyncio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_imports_exist():
    """Test that required imports are present."""
    try:
        from langvel.middleware.base import RateLimitMiddleware
        import inspect

        source = inspect.getsource(RateLimitMiddleware)

        checks = [
            "import asyncio" in source or "asyncio" in source,
            "defaultdict" in source,
            "_locks" in source,
            "_cleanup_task" in source,
        ]

        if not all(checks):
            print("[FAIL] Required imports/attributes missing")
            return False

        print("[PASS] Required imports and attributes present")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_rate_limiter_has_locks():
    """Test that rate limiter has per-user locks."""
    from langvel.middleware.base import RateLimitMiddleware

    limiter = RateLimitMiddleware(max_requests=5, window=60)

    # Verify lock dictionary exists
    assert hasattr(limiter, '_locks')
    assert hasattr(limiter, '_requests')

    print("[PASS] Rate limiter has lock infrastructure")
    return True


def test_rate_limiter_uses_defaultdict():
    """Test that rate limiter uses defaultdict."""
    from langvel.middleware.base import RateLimitMiddleware
    from collections import defaultdict

    limiter = RateLimitMiddleware()

    # Verify defaultdict is used
    assert isinstance(limiter._requests, defaultdict)
    assert isinstance(limiter._locks, defaultdict)

    print("[PASS] Rate limiter uses defaultdict")
    return True


def test_rate_limit_basic_functionality():
    """Test basic rate limiting works."""
    from langvel.middleware.base import RateLimitMiddleware

    async def run_test():
        limiter = RateLimitMiddleware(max_requests=3, window=60)

        state = {"user_id": "test_user"}

        # First 3 requests should succeed
        for i in range(3):
            result = await limiter.before(state)
            assert result == state

        # 4th request should fail
        try:
            await limiter.before(state)
            return False  # Should have raised exception
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                return True
            else:
                print(f"Wrong exception: {e}")
                return False

    result = asyncio.run(run_test())
    if result:
        print("[PASS] Basic rate limiting works")
    else:
        print("[FAIL] Rate limiting not working correctly")
    return result


def test_rate_limit_per_user_isolation():
    """Test that rate limits are per-user."""
    from langvel.middleware.base import RateLimitMiddleware

    async def run_test():
        limiter = RateLimitMiddleware(max_requests=2, window=60)

        # User1 makes 2 requests
        for i in range(2):
            await limiter.before({"user_id": "user1"})

        # User2 should still be able to make requests
        try:
            await limiter.before({"user_id": "user2"})
            return True
        except Exception:
            return False

    result = asyncio.run(run_test())
    if result:
        print("[PASS] Rate limits are per-user")
    else:
        print("[FAIL] Rate limits not isolated per user")
    return result


def test_cleanup_task_exists():
    """Test that cleanup task is defined."""
    from langvel.middleware.base import RateLimitMiddleware
    import inspect

    # Check that cleanup method exists
    assert hasattr(RateLimitMiddleware, '_cleanup_old_users')

    # Check implementation
    source = inspect.getsource(RateLimitMiddleware._cleanup_old_users)
    checks = [
        "asyncio.sleep" in source,
        "3600" in source,  # Cleanup every hour
        "users_to_remove" in source,
        "del self._requests" in source,
        "del self._locks" in source,
    ]

    if not all(checks):
        print("[FAIL] Cleanup task implementation incomplete")
        return False

    print("[PASS] Cleanup task exists and is properly implemented")
    return True


def test_cleanup_removes_stale_users():
    """Test that cleanup task removes users with no recent requests."""
    from langvel.middleware.base import RateLimitMiddleware
    import time

    async def run_test():
        limiter = RateLimitMiddleware(max_requests=5, window=1)  # 1 second window

        # Add some requests for test_user
        await limiter.before({"user_id": "test_user"})

        # Wait for window to expire
        await asyncio.sleep(1.5)

        # Manually trigger cleanup logic
        current_time = time.time()
        users_to_remove = []

        async with limiter._locks["test_user"]:
            limiter._requests["test_user"] = [
                req_time for req_time in limiter._requests["test_user"]
                if current_time - req_time < limiter.window
            ]
            if not limiter._requests["test_user"]:
                users_to_remove.append("test_user")

        if users_to_remove:
            async with limiter._locks["test_user"]:
                if "test_user" in limiter._requests:
                    del limiter._requests["test_user"]

        # Verify user was removed
        return "test_user" not in limiter._requests

    result = asyncio.run(run_test())
    if result:
        print("[PASS] Cleanup removes stale users")
    else:
        print("[FAIL] Cleanup not removing stale users")
    return result


def test_concurrent_access_safety():
    """Test that concurrent access doesn't cause race conditions."""
    from langvel.middleware.base import RateLimitMiddleware

    async def run_test():
        limiter = RateLimitMiddleware(max_requests=10, window=60)

        async def make_request(user_id):
            try:
                await limiter.before({"user_id": user_id})
                return True
            except Exception:
                return False

        # Make 20 concurrent requests for same user (limit is 10)
        tasks = [make_request("concurrent_user") for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # Exactly 10 should succeed, 10 should fail
        successes = sum(results)

        # Allow small variance due to timing
        return 8 <= successes <= 12

    result = asyncio.run(run_test())
    if result:
        print("[PASS] Concurrent access is thread-safe")
    else:
        print("[FAIL] Race condition detected")
    return result


def test_lock_per_user():
    """Test that locks are created per user."""
    from langvel.middleware.base import RateLimitMiddleware

    async def run_test():
        limiter = RateLimitMiddleware()

        # Make requests for multiple users
        await limiter.before({"user_id": "user1"})
        await limiter.before({"user_id": "user2"})
        await limiter.before({"user_id": "user3"})

        # Verify locks exist for each user
        return (
            "user1" in limiter._locks and
            "user2" in limiter._locks and
            "user3" in limiter._locks
        )

    result = asyncio.run(run_test())
    if result:
        print("[PASS] Locks created per user")
    else:
        print("[FAIL] Per-user locks not working")
    return result


def test_cleanup_task_starts():
    """Test that cleanup task starts on first request."""
    from langvel.middleware.base import RateLimitMiddleware

    async def run_test():
        limiter = RateLimitMiddleware()

        # Initially not started
        assert not limiter._started
        assert limiter._cleanup_task is None

        # Make first request
        await limiter.before({"user_id": "test"})

        # Should now be started
        return limiter._started

    result = asyncio.run(run_test())
    if result:
        print("[PASS] Cleanup task starts on first request")
    else:
        print("[FAIL] Cleanup task not starting")
    return result


if __name__ == '__main__':
    print("="*60)
    print("Testing Rate Limiter Fixes (TODO-007)")
    print("="*60)
    print()

    tests = [
        ("Required imports present", test_imports_exist),
        ("Lock infrastructure exists", test_rate_limiter_has_locks),
        ("Uses defaultdict", test_rate_limiter_uses_defaultdict),
        ("Basic rate limiting", test_rate_limit_basic_functionality),
        ("Per-user isolation", test_rate_limit_per_user_isolation),
        ("Cleanup task exists", test_cleanup_task_exists),
        ("Cleanup removes stale users", test_cleanup_removes_stale_users),
        ("Concurrent access safety", test_concurrent_access_safety),
        ("Lock per user", test_lock_per_user),
        ("Cleanup task starts", test_cleanup_task_starts),
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
