"""Agent routing system - Laravel-inspired route definitions."""

from typing import Any, Callable, Dict, List, Optional, Type
from langvel.core.agent import Agent


class AgentRouter:
    """
    Main router for registering agent flows.

    Similar to Laravel's routing system, this allows you to define
    agent flows with a clean, declarative syntax.

    Example:
        router = AgentRouter()

        @router.flow('/customer-support')
        class CustomerSupportAgent(Agent):
            def build_graph(self):
                return self.start().then(self.handle).end()
    """

    def __init__(self):
        self._routes: Dict[str, Type[Agent]] = {}
        self._middleware: Dict[str, List[str]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def flow(
        self,
        path: str,
        middleware: Optional[List[str]] = None,
        name: Optional[str] = None,
        **metadata
    ) -> Callable:
        """
        Register an agent flow.

        Args:
            path: The route path (e.g., '/customer-support')
            middleware: List of middleware to apply
            name: Optional name for the route
            **metadata: Additional metadata

        Returns:
            Decorator function
        """
        def decorator(agent_class: Type[Agent]) -> Type[Agent]:
            # Register the route
            route_name = name or agent_class.__name__
            self._routes[path] = agent_class
            self._middleware[path] = middleware or []
            self._metadata[path] = {
                'name': route_name,
                **metadata
            }

            # Add middleware to agent class if specified
            if middleware:
                if not hasattr(agent_class, 'middleware'):
                    agent_class.middleware = []
                agent_class.middleware.extend(middleware)

            return agent_class

        return decorator

    def get(self, path: str) -> Optional[Type[Agent]]:
        """
        Get an agent class by path.

        Args:
            path: The route path

        Returns:
            Agent class if found, None otherwise
        """
        return self._routes.get(path)

    def get_all(self) -> Dict[str, Type[Agent]]:
        """Get all registered routes."""
        return self._routes.copy()

    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get metadata for a route."""
        return self._metadata.get(path, {})

    def list_routes(self) -> List[Dict[str, Any]]:
        """
        List all registered routes with their metadata.

        Returns:
            List of route information
        """
        routes = []
        for path, agent_class in self._routes.items():
            routes.append({
                'path': path,
                'agent': agent_class.__name__,
                'middleware': self._middleware.get(path, []),
                'metadata': self._metadata.get(path, {})
            })
        return routes

    def group(
        self,
        prefix: str = "",
        middleware: Optional[List[str]] = None,
        **group_metadata
    ) -> "RouteGroup":
        """
        Create a route group with shared configuration.

        Args:
            prefix: Path prefix for all routes in the group
            middleware: Shared middleware for all routes
            **group_metadata: Shared metadata

        Returns:
            RouteGroup instance

        Example:
            with router.group(prefix='/admin', middleware=['auth', 'admin']):
                @router.flow('/dashboard')
                class AdminDashboard(Agent):
                    pass
        """
        return RouteGroup(self, prefix, middleware or [], group_metadata)


class RouteGroup:
    """
    Route group for organizing related flows.

    Similar to Laravel's route groups.
    """

    def __init__(
        self,
        router: AgentRouter,
        prefix: str,
        middleware: List[str],
        metadata: Dict[str, Any]
    ):
        self.router = router
        self.prefix = prefix
        self.middleware = middleware
        self.metadata = metadata
        self._original_flow = router.flow

    def __enter__(self):
        """Enter the route group context."""
        # Patch the router's flow method to include group settings
        original_flow = self.router.flow

        def grouped_flow(path: str, middleware: Optional[List[str]] = None, **kwargs):
            # Combine group and route settings
            full_path = f"{self.prefix}{path}"
            combined_middleware = [*self.middleware, *(middleware or [])]
            combined_metadata = {**self.metadata, **kwargs}

            return original_flow(full_path, combined_middleware, **combined_metadata)

        self.router.flow = grouped_flow
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the route group context."""
        # Restore original flow method
        self.router.flow = self._original_flow


# Global router instance
router = AgentRouter()
