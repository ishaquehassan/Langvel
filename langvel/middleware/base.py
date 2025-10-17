"""Middleware system - Laravel-inspired middleware for agents."""

from typing import Any, Callable, Dict
from abc import ABC, abstractmethod


class Middleware(ABC):
    """
    Base middleware class.

    Similar to Laravel's middleware, this allows you to inspect and modify
    the state before and after agent execution.

    Example:
        class AuthMiddleware(Middleware):
            async def before(self, state: dict) -> dict:
                # Verify authentication
                if not state.get('user_id'):
                    raise AuthenticationError("Not authenticated")
                return state

            async def after(self, state: dict) -> dict:
                # Log the result
                logger.info(f"Request completed for user {state.get('user_id')}")
                return state
    """

    @abstractmethod
    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute before the agent runs.

        Args:
            state: The input state

        Returns:
            Modified state

        Raises:
            Exception: If middleware wants to block execution
        """
        pass

    @abstractmethod
    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute after the agent runs.

        Args:
            state: The output state

        Returns:
            Modified state
        """
        pass


class RateLimitMiddleware(Middleware):
    """Rate limiting middleware."""

    def __init__(self, max_requests: int = 10, window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed
            window: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = window
        self._requests: Dict[str, list] = {}

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit before execution."""
        import time

        user_id = state.get('user_id', 'anonymous')
        current_time = time.time()

        # Initialize user tracking
        if user_id not in self._requests:
            self._requests[user_id] = []

        # Clean old requests
        self._requests[user_id] = [
            req_time for req_time in self._requests[user_id]
            if current_time - req_time < self.window
        ]

        # Check limit
        if len(self._requests[user_id]) >= self.max_requests:
            raise Exception(f"Rate limit exceeded: {self.max_requests} requests per {self.window}s")

        # Add current request
        self._requests[user_id].append(current_time)

        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nothing to do after execution."""
        return state


class LoggingMiddleware(Middleware):
    """Logging middleware."""

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Log before execution."""
        import json
        print(f"[Langvel] Input State: {json.dumps(state, indent=2, default=str)}")
        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Log after execution."""
        import json
        print(f"[Langvel] Output State: {json.dumps(state, indent=2, default=str)}")
        return state


class AuthenticationMiddleware(Middleware):
    """Authentication middleware."""

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify authentication."""
        auth_state = state.get('auth_state', 'guest')

        if auth_state != 'authenticated':
            raise Exception("Authentication required")

        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nothing to do after execution."""
        return state


class ValidationMiddleware(Middleware):
    """State validation middleware."""

    def __init__(self, required_fields: list):
        """
        Initialize validator.

        Args:
            required_fields: List of required field names
        """
        self.required_fields = required_fields

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate required fields."""
        missing = [field for field in self.required_fields if field not in state]

        if missing:
            raise Exception(f"Missing required fields: {', '.join(missing)}")

        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nothing to do after execution."""
        return state


class CORSMiddleware(Middleware):
    """CORS headers middleware."""

    def __init__(self, allowed_origins: list = None):
        """
        Initialize CORS middleware.

        Args:
            allowed_origins: List of allowed origins
        """
        self.allowed_origins = allowed_origins or ['*']

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nothing to do before execution."""
        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Add CORS headers to metadata."""
        if 'metadata' not in state:
            state['metadata'] = {}

        state['metadata']['cors_headers'] = {
            'Access-Control-Allow-Origin': self.allowed_origins[0],
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }

        return state
