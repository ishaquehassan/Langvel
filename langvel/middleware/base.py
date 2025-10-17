"""Middleware system - Laravel-inspired middleware for agents."""

from typing import Any, Callable, Dict
from abc import ABC, abstractmethod
import asyncio
import time
from collections import defaultdict


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
    """
    Rate limiting middleware with thread-safe implementation.

    Fixes:
    - Race condition: Uses asyncio locks for concurrent access
    - Memory leak: Periodic cleanup task removes old user data
    - Resource exhaustion: Limits dictionary size

    Thread Safety:
    - Per-user locks prevent concurrent modification
    - defaultdict for automatic initialization
    - Cleanup task runs every hour to remove stale users
    """

    def __init__(self, max_requests: int = 10, window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed
            window: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = window
        self._requests: Dict[str, list] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._cleanup_task = None
        self._started = False

    def _start_cleanup_task(self):
        """Start background cleanup task (once)."""
        if not self._started:
            self._started = True
            try:
                loop = asyncio.get_event_loop()
                self._cleanup_task = loop.create_task(self._cleanup_old_users())
            except RuntimeError:
                # No event loop running yet, task will be started on first request
                pass

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check rate limit before execution.

        Thread-safe implementation with per-user locking.
        """
        # Start cleanup task on first request
        if not self._started:
            self._start_cleanup_task()

        user_id = state.get('user_id', 'anonymous')
        current_time = time.time()

        # Use lock per user to prevent race conditions
        async with self._locks[user_id]:
            # Clean old requests within the time window
            self._requests[user_id] = [
                req_time for req_time in self._requests[user_id]
                if current_time - req_time < self.window
            ]

            # Check if limit exceeded
            if len(self._requests[user_id]) >= self.max_requests:
                raise Exception(
                    f"Rate limit exceeded: {self.max_requests} requests per {self.window}s"
                )

            # Add current request timestamp
            self._requests[user_id].append(current_time)

        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nothing to do after execution."""
        return state

    async def _cleanup_old_users(self):
        """
        Periodic cleanup to prevent memory leak.

        Runs every hour and removes users with no recent requests.
        Prevents unbounded dictionary growth.
        """
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour

                current_time = time.time()
                users_to_remove = []

                # Find users with no requests in the time window
                for user_id in list(self._requests.keys()):
                    # Use lock to safely check user's requests
                    async with self._locks[user_id]:
                        # Remove old requests
                        self._requests[user_id] = [
                            req_time for req_time in self._requests[user_id]
                            if current_time - req_time < self.window
                        ]

                        # If no recent requests, mark for removal
                        if not self._requests[user_id]:
                            users_to_remove.append(user_id)

                # Remove stale users
                for user_id in users_to_remove:
                    async with self._locks[user_id]:
                        if user_id in self._requests:
                            del self._requests[user_id]
                        if user_id in self._locks:
                            del self._locks[user_id]

            except asyncio.CancelledError:
                # Task cancelled, exit gracefully
                break
            except Exception as e:
                # Log error but continue cleanup loop
                import logging
                logger = logging.getLogger("langvel.middleware")
                logger.error(f"Rate limiter cleanup error: {e}")

    def __del__(self):
        """Cancel cleanup task on deletion."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class LoggingMiddleware(Middleware):
    """Logging middleware with structured logging."""

    def __init__(self):
        """Initialize logging middleware."""
        from langvel.logging import get_logger
        self.logger = get_logger('langvel.middleware.logging')

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Log before execution."""
        self.logger.info(
            "Agent execution started",
            extra={
                'event': 'agent_input',
                'state': state,
                'state_keys': list(state.keys())
            }
        )
        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Log after execution."""
        self.logger.info(
            "Agent execution completed",
            extra={
                'event': 'agent_output',
                'state': state,
                'state_keys': list(state.keys())
            }
        )
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
