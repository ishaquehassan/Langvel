# -*- coding: utf-8 -*-
"""Test JWT secret key validation."""

import pytest
import os
from langvel.auth.manager import AuthManager


def test_jwt_secret_required():
    """Test that AuthManager requires JWT_SECRET_KEY in environment."""
    # Remove JWT_SECRET_KEY from environment if present
    old_key = os.environ.pop('JWT_SECRET_KEY', None)

    try:
        # Should raise RuntimeError when JWT_SECRET_KEY is not set
        with pytest.raises(RuntimeError) as exc_info:
            AuthManager()

        # Verify error message is helpful
        error_message = str(exc_info.value)
        assert "JWT_SECRET_KEY" in error_message
        assert "required" in error_message.lower()
        assert "secrets.token_urlsafe" in error_message

    finally:
        # Restore old key if it existed
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key


def test_jwt_secret_from_environment():
    """Test that AuthManager uses JWT_SECRET_KEY from environment."""
    # Set a test secret key
    test_secret = "test_secret_key_12345"
    old_key = os.environ.get('JWT_SECRET_KEY')

    try:
        os.environ['JWT_SECRET_KEY'] = test_secret

        # Should successfully create AuthManager
        auth = AuthManager()

        # Verify it uses the environment variable
        assert auth.secret_key == test_secret

        # Verify tokens work with this secret
        token = auth.create_token(user_id="test_user", permissions=["read"])
        assert token is not None

        # Verify token can be decoded
        payload = auth.verify_token(token)
        assert payload['user_id'] == "test_user"
        assert payload['permissions'] == ["read"]

    finally:
        # Restore old key
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key
        else:
            os.environ.pop('JWT_SECRET_KEY', None)


def test_jwt_secret_explicit_parameter():
    """Test that explicit secret_key parameter works."""
    test_secret = "explicit_secret_key"

    # Should work even without environment variable
    old_key = os.environ.pop('JWT_SECRET_KEY', None)

    try:
        auth = AuthManager(secret_key=test_secret)
        assert auth.secret_key == test_secret

        # Verify tokens work
        token = auth.create_token(user_id="test_user")
        payload = auth.verify_token(token)
        assert payload['user_id'] == "test_user"

    finally:
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key


def test_tokens_consistent_across_restarts():
    """Test that tokens work across 'restarts' with same secret."""
    test_secret = "persistent_secret_key"
    old_key = os.environ.get('JWT_SECRET_KEY')

    try:
        os.environ['JWT_SECRET_KEY'] = test_secret

        # Create first manager and token
        auth1 = AuthManager()
        token = auth1.create_token(user_id="user1", permissions=["admin"])

        # Simulate restart - create new manager instance
        auth2 = AuthManager()

        # Token created by auth1 should be valid for auth2
        payload = auth2.verify_token(token)
        assert payload['user_id'] == "user1"
        assert payload['permissions'] == ["admin"]

    finally:
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key
        else:
            os.environ.pop('JWT_SECRET_KEY', None)


def test_tokens_fail_with_different_secret():
    """Test that tokens from different secrets are invalid."""
    old_key = os.environ.get('JWT_SECRET_KEY')

    try:
        # Create token with first secret
        os.environ['JWT_SECRET_KEY'] = "secret_key_1"
        auth1 = AuthManager()
        token = auth1.create_token(user_id="user1")

        # Try to verify with different secret (simulates different worker)
        os.environ['JWT_SECRET_KEY'] = "secret_key_2"
        auth2 = AuthManager()

        # Should fail
        from langvel.auth.manager import AuthenticationError
        with pytest.raises(AuthenticationError) as exc_info:
            auth2.verify_token(token)

        assert "Invalid token" in str(exc_info.value)

    finally:
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key
        else:
            os.environ.pop('JWT_SECRET_KEY', None)


if __name__ == '__main__':
    """Run tests manually."""
    import sys

    print("Testing JWT Secret Key Validation...\n")

    tests = [
        test_jwt_secret_required,
        test_jwt_secret_from_environment,
        test_jwt_secret_explicit_parameter,
        test_tokens_consistent_across_restarts,
        test_tokens_fail_with_different_secret,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}")
            print(f"  Error: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print(f"{'='*50}")

    if failed > 0:
        sys.exit(1)
