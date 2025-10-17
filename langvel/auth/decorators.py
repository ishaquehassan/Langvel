"""Authentication decorators for agents."""

from typing import Callable
from functools import wraps


def requires_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication for an agent node.

    Example:
        @requires_auth
        async def handle_sensitive_data(self, state):
            # Only authenticated users can reach this
            pass
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get state from args (typically the second argument)
        state = args[1] if len(args) > 1 else kwargs.get('state')

        if state is None:
            raise ValueError("State not found in function arguments")

        # Check authentication
        auth_state = getattr(state, 'auth_state', 'guest')
        if auth_state != 'authenticated':
            raise PermissionError("Authentication required")

        return await func(*args, **kwargs)

    wrapper._requires_auth = True
    return wrapper


def requires_permission(permission: str) -> Callable:
    """
    Decorator to require a specific permission.

    Args:
        permission: Permission name required

    Example:
        @requires_permission('admin')
        async def admin_action(self, state):
            # Only users with 'admin' permission can reach this
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            state = args[1] if len(args) > 1 else kwargs.get('state')

            if state is None:
                raise ValueError("State not found in function arguments")

            # Check permission
            permissions = getattr(state, 'permissions', [])
            if permission not in permissions:
                raise PermissionError(f"Permission '{permission}' required")

            return await func(*args, **kwargs)

        wrapper._requires_permission = permission
        return wrapper

    return decorator


def rate_limit(max_requests: int = 10, window: int = 60) -> Callable:
    """
    Decorator to add rate limiting to an agent node.

    Args:
        max_requests: Maximum requests allowed
        window: Time window in seconds

    Example:
        @rate_limit(max_requests=5, window=60)
        async def expensive_operation(self, state):
            # Rate limited to 5 requests per minute
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Store request history per user
        request_history = {}

        @wraps(func)
        async def wrapper(*args, **kwargs):
            import time

            state = args[1] if len(args) > 1 else kwargs.get('state')
            user_id = getattr(state, 'user_id', 'anonymous')
            current_time = time.time()

            # Initialize tracking
            if user_id not in request_history:
                request_history[user_id] = []

            # Clean old requests
            request_history[user_id] = [
                req_time for req_time in request_history[user_id]
                if current_time - req_time < window
            ]

            # Check limit
            if len(request_history[user_id]) >= max_requests:
                raise Exception(f"Rate limit exceeded: {max_requests} requests per {window}s")

            # Add current request
            request_history[user_id].append(current_time)

            return await func(*args, **kwargs)

        wrapper._rate_limit = {'max_requests': max_requests, 'window': window}
        return wrapper

    return decorator


def validate_state(schema_class) -> Callable:
    """
    Decorator to validate state using a Pydantic model.

    Args:
        schema_class: Pydantic model class for validation

    Example:
        @validate_state(CustomerRequestSchema)
        async def process_request(self, state):
            # State is validated against schema
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            state = args[1] if len(args) > 1 else kwargs.get('state')

            # Validate state
            try:
                validated = schema_class(**state.to_dict() if hasattr(state, 'to_dict') else state)
            except Exception as e:
                raise ValueError(f"State validation failed: {str(e)}")

            return await func(*args, **kwargs)

        wrapper._validate_state = schema_class
        return wrapper

    return decorator
