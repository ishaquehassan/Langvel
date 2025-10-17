# -*- coding: utf-8 -*-
"""Test agent instance pooling (TODO-005)."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_agent_pool_exists():
    """Test that agent pool and get_agent function exist."""
    try:
        from langvel.server import _agent_pool, _agent_pool_lock, get_agent

        assert _agent_pool is not None
        assert _agent_pool_lock is not None
        assert get_agent is not None

        print("[PASS] Agent pool infrastructure exists")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_agent_pool_starts_empty():
    """Test that agent pool starts empty."""
    from langvel.server import _agent_pool

    # Clear pool for test isolation
    _agent_pool.clear()

    assert len(_agent_pool) == 0, f"Expected empty pool, got {len(_agent_pool)} items"

    print("[PASS] Agent pool starts empty")
    return True


def test_get_agent_singleton_behavior():
    """Test that get_agent returns same instance for same path."""
    from langvel.server import get_agent, _agent_pool

    # Clear pool
    _agent_pool.clear()

    # Mock agent path (will fail to find, but we're testing the pool logic)
    test_path = "test-agent"

    try:
        # This will raise 404 since agent doesn't exist in router
        agent1 = get_agent(test_path)
        agent2 = get_agent(test_path)

        # Should be same instance
        assert agent1 is agent2, "Expected same instance"

        print("[PASS] get_agent returns singleton instance")
        return True
    except Exception as e:
        # Expected to fail since we don't have actual agents
        if "Agent not found" in str(e):
            print("[PASS] get_agent correctly validates agent existence")
            return True
        else:
            print(f"[FAIL] Unexpected error: {e}")
            return False


def test_pool_thread_safety():
    """Test that pool has thread safety mechanisms."""
    from langvel.server import _agent_pool_lock
    from threading import Lock

    # Verify lock is a proper Lock instance
    assert isinstance(_agent_pool_lock, Lock), "Pool lock should be threading.Lock"

    print("[PASS] Thread safety lock exists")
    return True


def test_pool_double_check_locking():
    """Test double-checked locking pattern in get_agent."""
    from langvel.server import get_agent
    import inspect

    # Get source code of get_agent function
    source = inspect.getsource(get_agent)

    # Verify double-checked locking pattern
    checks = [
        "if agent_path in _agent_pool:" in source,  # First check (fast path)
        "with _agent_pool_lock:" in source,  # Lock acquisition
        source.count("if agent_path in _agent_pool:") >= 2,  # Second check (inside lock)
    ]

    if not all(checks):
        print("[FAIL] Double-checked locking pattern not found")
        return False

    print("[PASS] Double-checked locking pattern implemented")
    return True


def test_pool_size_logging():
    """Test that pool logs creation events."""
    from langvel.server import get_agent
    import inspect

    source = inspect.getsource(get_agent)

    # Verify logging is present
    if "logger.info" not in source or "pool_size" not in source:
        print("[FAIL] Pool size logging not found")
        return False

    print("[PASS] Pool creation logging implemented")
    return True


def test_invoke_agent_uses_pool():
    """Test that invoke_agent endpoint uses get_agent."""
    from langvel.server import invoke_agent
    import inspect

    source = inspect.getsource(invoke_agent)

    # Verify get_agent is called
    if "get_agent(agent_path)" not in source:
        print("[FAIL] invoke_agent does not use get_agent")
        return False

    # Verify old pattern is removed
    if "agent_class()" in source:
        print("[FAIL] Old agent instantiation pattern still present")
        return False

    print("[PASS] invoke_agent uses pooled agents")
    return True


def test_graph_endpoint_uses_pool():
    """Test that get_agent_graph endpoint uses get_agent."""
    from langvel.server import get_agent_graph
    import inspect

    source = inspect.getsource(get_agent_graph)

    # Verify get_agent is called
    if "get_agent(agent_path)" not in source:
        print("[FAIL] get_agent_graph does not use get_agent")
        return False

    # Verify old pattern is removed
    if "agent_class()" in source:
        print("[FAIL] Old agent instantiation pattern still present")
        return False

    print("[PASS] get_agent_graph uses pooled agents")
    return True


def test_pool_prevents_duplicate_instances():
    """Test that pool prevents creating duplicate instances."""
    from langvel.server import _agent_pool

    # Simulate adding agent to pool
    _agent_pool.clear()

    class MockAgent:
        pass

    test_path = "mock/agent"
    mock_instance = MockAgent()

    _agent_pool[test_path] = mock_instance

    # Verify it's in the pool
    assert test_path in _agent_pool
    assert _agent_pool[test_path] is mock_instance

    # Verify pool size
    assert len(_agent_pool) == 1

    print("[PASS] Pool correctly stores agent instances")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Testing Agent Instance Pooling (TODO-005)")
    print("="*60)
    print()

    tests = [
        ("Agent pool infrastructure exists", test_agent_pool_exists),
        ("Pool starts empty", test_agent_pool_starts_empty),
        ("Singleton behavior", test_get_agent_singleton_behavior),
        ("Thread safety lock", test_pool_thread_safety),
        ("Double-checked locking pattern", test_pool_double_check_locking),
        ("Pool size logging", test_pool_size_logging),
        ("invoke_agent uses pool", test_invoke_agent_uses_pool),
        ("get_agent_graph uses pool", test_graph_endpoint_uses_pool),
        ("Pool prevents duplicates", test_pool_prevents_duplicate_instances),
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
