"""Middleware manager - Orchestrates middleware execution."""

from typing import Any, Dict, List, Type
from langvel.middleware.base import Middleware


class MiddlewareManager:
    """
    Manages middleware execution pipeline.

    Handles registration and execution of middleware in the correct order.
    """

    # Registry of available middleware
    _middleware_registry: Dict[str, Type[Middleware]] = {}

    def __init__(self):
        self._middleware_stack: List[Middleware] = []

    @classmethod
    def register_middleware(cls, name: str, middleware_class: Type[Middleware]) -> None:
        """
        Register a middleware class globally.

        Args:
            name: Middleware identifier
            middleware_class: Middleware class
        """
        cls._middleware_registry[name] = middleware_class

    def register(self, middleware: str | Type[Middleware] | Middleware) -> None:
        """
        Register middleware to this manager instance.

        Args:
            middleware: Middleware name (str), class, or instance
        """
        if isinstance(middleware, str):
            # Look up by name
            if middleware not in self._middleware_registry:
                raise ValueError(f"Middleware '{middleware}' not found in registry")
            middleware_class = self._middleware_registry[middleware]
            middleware_instance = middleware_class()

        elif isinstance(middleware, type) and issubclass(middleware, Middleware):
            # Instantiate the class
            middleware_instance = middleware()

        elif isinstance(middleware, Middleware):
            # Use the instance directly
            middleware_instance = middleware

        else:
            raise TypeError(f"Invalid middleware type: {type(middleware)}")

        self._middleware_stack.append(middleware_instance)

    async def run_before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all middleware before hooks.

        Args:
            state: Input state

        Returns:
            Modified state after all middleware

        Raises:
            Exception: If any middleware blocks execution
        """
        for middleware in self._middleware_stack:
            state = await middleware.before(state)
        return state

    async def run_after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all middleware after hooks in reverse order.

        Args:
            state: Output state

        Returns:
            Modified state after all middleware
        """
        # Run in reverse order (LIFO)
        for middleware in reversed(self._middleware_stack):
            state = await middleware.after(state)
        return state

    def clear(self) -> None:
        """Clear all registered middleware."""
        self._middleware_stack.clear()


# Register built-in middleware
from langvel.middleware.base import (
    RateLimitMiddleware,
    LoggingMiddleware,
    AuthenticationMiddleware,
    ValidationMiddleware,
    CORSMiddleware
)

MiddlewareManager.register_middleware('rate_limit', RateLimitMiddleware)
MiddlewareManager.register_middleware('logging', LoggingMiddleware)
MiddlewareManager.register_middleware('auth', AuthenticationMiddleware)
MiddlewareManager.register_middleware('validation', ValidationMiddleware)
MiddlewareManager.register_middleware('cors', CORSMiddleware)
