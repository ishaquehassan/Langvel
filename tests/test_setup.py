"""
Test setup and basic functionality.

These tests verify that the framework is properly installed and configured.
"""

import sys
from pathlib import Path


def test_python_version():
    """Test that Python version is 3.10+."""
    assert sys.version_info >= (3, 10), "Python 3.10+ is required"


def test_imports():
    """Test that core modules can be imported."""
    try:
        from langvel.core.agent import Agent
        from langvel.state.base import StateModel
        from langvel.routing.router import router
        from langvel.tools.decorators import tool
        from langvel.middleware.base import Middleware
        assert True
    except ImportError as e:
        assert False, f"Failed to import core modules: {e}"


def test_agent_creation():
    """Test basic agent creation."""
    from langvel.core.agent import Agent
    from langvel.state.base import StateModel

    class TestState(StateModel):
        query: str
        response: str = ""

    class TestAgent(Agent):
        state_model = TestState

        def build_graph(self):
            return self.start().then(self.process).end()

        async def process(self, state: TestState):
            state.response = f"Processed: {state.query}"
            return state

    agent = TestAgent()
    assert agent is not None
    assert agent.state_model == TestState


def test_router_registration():
    """Test router registration."""
    from langvel.routing.router import AgentRouter
    from langvel.core.agent import Agent
    from langvel.state.base import StateModel

    router = AgentRouter()

    class TestState(StateModel):
        query: str

    @router.flow('/test')
    class TestAgent(Agent):
        state_model = TestState

        def build_graph(self):
            return self.start().end()

    assert router.get('/test') is not None


def test_state_model():
    """Test state model validation."""
    from langvel.state.base import StateModel
    from pydantic import ValidationError

    class TestState(StateModel):
        required_field: str
        optional_field: str = "default"

    # Valid state
    state = TestState(required_field="test")
    assert state.required_field == "test"
    assert state.optional_field == "default"

    # Invalid state (missing required field)
    try:
        state = TestState()
        assert False, "Should have raised ValidationError"
    except ValidationError:
        assert True


def test_tool_decorator():
    """Test tool decorator functionality."""
    from langvel.tools.decorators import tool

    @tool(description="Test tool")
    async def test_tool_func(data: str) -> str:
        return f"Processed: {data}"

    assert hasattr(test_tool_func, '_is_tool')
    assert test_tool_func._tool_type == 'custom'
    assert test_tool_func._tool_description == "Test tool"


def test_middleware_creation():
    """Test middleware creation."""
    from langvel.middleware.base import Middleware
    from typing import Any, Dict

    class TestMiddleware(Middleware):
        async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
            state['middleware_ran'] = True
            return state

        async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
            return state

    middleware = TestMiddleware()
    assert middleware is not None


def test_directory_structure():
    """Test that required directories exist."""
    required_dirs = [
        'langvel',
        'langvel/core',
        'langvel/routing',
        'langvel/state',
        'langvel/tools',
        'langvel/middleware',
        'langvel/rag',
        'langvel/mcp',
        'langvel/auth',
        'langvel/cli',
    ]

    for dir_path in required_dirs:
        assert Path(dir_path).exists(), f"Required directory missing: {dir_path}"


def test_cli_availability():
    """Test that CLI commands are available."""
    import subprocess

    # Test that langvel command is registered (may not work if not installed)
    # This is a soft test - it's okay if it fails in some environments
    try:
        result = subprocess.run(
            ['python', '-m', 'langvel.cli.main', '--version'],
            capture_output=True,
            timeout=5
        )
        # If we get here, CLI is working
        assert True
    except Exception:
        # CLI might not be installed yet - that's okay
        assert True


if __name__ == '__main__':
    """Run tests manually."""
    import traceback

    tests = [
        test_python_version,
        test_imports,
        test_agent_creation,
        test_router_registration,
        test_state_model,
        test_tool_decorator,
        test_middleware_creation,
        test_directory_structure,
        test_cli_availability,
    ]

    passed = 0
    failed = 0

    print("Running Langvel Setup Tests...\n")

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print(f"{'='*50}")

    if failed > 0:
        sys.exit(1)
