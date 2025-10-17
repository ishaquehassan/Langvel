"""Authentication manager - JWT and API key support."""

from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import hashlib
import secrets
import jwt


class AuthenticationError(Exception):
    """Exception raised for authentication failures."""
    pass


class AuthManager:
    """
    Manages authentication via JWT tokens and API keys.

    Supports:
    - JWT token generation and validation
    - API key management
    - Role-based access control (RBAC)
    - Session management
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expiry: int = 3600
    ):
        """
        Initialize authentication manager.

        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm (default: HS256)
            token_expiry: Token expiry in seconds (default: 1 hour)
        """
        self.secret_key = secret_key or self._get_secret_key()
        self.algorithm = algorithm
        self.token_expiry = token_expiry
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _get_secret_key(self) -> str:
        """Get secret key from environment."""
        import os
        return os.getenv('JWT_SECRET_KEY') or secrets.token_urlsafe(32)

    # JWT Token Methods

    def create_token(
        self,
        user_id: str,
        permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a JWT token.

        Args:
            user_id: User identifier
            permissions: List of permissions
            metadata: Additional metadata

        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'iat': now,
            'exp': now + timedelta(seconds=self.token_expiry),
            'metadata': metadata or {}
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")

    def refresh_token(self, token: str) -> str:
        """
        Refresh an existing token.

        Args:
            token: Existing JWT token

        Returns:
            New JWT token

        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Verify existing token (but allow expired for refresh)
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )

            # Create new token with same claims
            return self.create_token(
                user_id=payload['user_id'],
                permissions=payload.get('permissions', []),
                metadata=payload.get('metadata', {})
            )

        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Cannot refresh invalid token: {str(e)}")

    # API Key Methods

    def create_api_key(
        self,
        name: str,
        permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an API key.

        Args:
            name: API key name/description
            permissions: List of permissions
            metadata: Additional metadata

        Returns:
            API key string
        """
        # Generate secure random key
        api_key = f"lv_{secrets.token_urlsafe(32)}"

        # Hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Store metadata
        self._api_keys[key_hash] = {
            'name': name,
            'permissions': permissions or [],
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat(),
            'last_used': None,
            'usage_count': 0
        }

        return api_key

    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Verify an API key.

        Args:
            api_key: API key string

        Returns:
            API key metadata

        Raises:
            AuthenticationError: If API key is invalid
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in self._api_keys:
            raise AuthenticationError("Invalid API key")

        # Update usage stats
        key_data = self._api_keys[key_hash]
        key_data['last_used'] = datetime.utcnow().isoformat()
        key_data['usage_count'] += 1

        return key_data

    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: API key string

        Returns:
            True if revoked, False if not found
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash in self._api_keys:
            del self._api_keys[key_hash]
            return True

        return False

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys (without revealing actual keys).

        Returns:
            List of API key metadata
        """
        return list(self._api_keys.values())

    # Session Management

    def create_session(
        self,
        user_id: str,
        permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a session.

        Args:
            user_id: User identifier
            permissions: List of permissions
            metadata: Additional metadata

        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)

        self._sessions[session_id] = {
            'user_id': user_id,
            'permissions': permissions or [],
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat()
        }

        return session_id

    def verify_session(self, session_id: str) -> Dict[str, Any]:
        """
        Verify a session.

        Args:
            session_id: Session ID

        Returns:
            Session data

        Raises:
            AuthenticationError: If session is invalid or expired
        """
        if session_id not in self._sessions:
            raise AuthenticationError("Invalid session")

        session = self._sessions[session_id]

        # Update last activity
        session['last_activity'] = datetime.utcnow().isoformat()

        return session

    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a session.

        Args:
            session_id: Session ID

        Returns:
            True if revoked, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True

        return False

    # Permission Checking

    def has_permission(
        self,
        user_permissions: List[str],
        required_permission: str
    ) -> bool:
        """
        Check if user has required permission.

        Supports wildcard permissions (e.g., 'admin.*' allows 'admin.read', 'admin.write').

        Args:
            user_permissions: List of user's permissions
            required_permission: Permission to check

        Returns:
            True if user has permission
        """
        # Check direct match
        if required_permission in user_permissions:
            return True

        # Check wildcard permissions
        for perm in user_permissions:
            if perm.endswith('.*'):
                prefix = perm[:-2]
                if required_permission.startswith(f"{prefix}."):
                    return True

        return False

    def has_all_permissions(
        self,
        user_permissions: List[str],
        required_permissions: List[str]
    ) -> bool:
        """
        Check if user has all required permissions.

        Args:
            user_permissions: List of user's permissions
            required_permissions: List of required permissions

        Returns:
            True if user has all permissions
        """
        return all(
            self.has_permission(user_permissions, perm)
            for perm in required_permissions
        )

    def has_any_permission(
        self,
        user_permissions: List[str],
        required_permissions: List[str]
    ) -> bool:
        """
        Check if user has any of the required permissions.

        Args:
            user_permissions: List of user's permissions
            required_permissions: List of required permissions

        Returns:
            True if user has at least one permission
        """
        return any(
            self.has_permission(user_permissions, perm)
            for perm in required_permissions
        )


# Global auth manager instance
_auth_manager = None


def get_auth_manager() -> AuthManager:
    """Get or create global auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
