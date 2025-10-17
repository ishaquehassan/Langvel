# -*- coding: utf-8 -*-
"""Simple test for JWT secret key validation (no pytest required)."""

import os
import sys

def test_jwt_secret_required():
    """Test that AuthManager requires JWT_SECRET_KEY in environment."""
    from langvel.auth.manager import AuthManager

    # Remove JWT_SECRET_KEY from environment if present
    old_key = os.environ.pop('JWT_SECRET_KEY', None)

    try:
        # Should raise RuntimeError when JWT_SECRET_KEY is not set
        try:
            AuthManager()
            print("[FAIL] AuthManager should raise RuntimeError when JWT_SECRET_KEY not set")
            return False
        except RuntimeError as e:
            error_message = str(e)
            if "JWT_SECRET_KEY" not in error_message or "required" not in error_message.lower():
                print(f"[FAIL] Error message not helpful: {error_message}")
                return False
            print("[PASS] AuthManager correctly requires JWT_SECRET_KEY")
            return True
    finally:
        # Restore old key if it existed
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key


def test_jwt_secret_from_environment():
    """Test that AuthManager uses JWT_SECRET_KEY from environment."""
    from langvel.auth.manager import AuthManager

    test_secret = "test_secret_key_12345"
    old_key = os.environ.get('JWT_SECRET_KEY')

    try:
        os.environ['JWT_SECRET_KEY'] = test_secret

        # Should successfully create AuthManager
        auth = AuthManager()

        # Verify it uses the environment variable
        if auth.secret_key != test_secret:
            print(f"[FAIL] Secret key mismatch: expected {test_secret}, got {auth.secret_key}")
            return False

        # Verify tokens work with this secret
        token = auth.create_token(user_id="test_user", permissions=["read"])

        # Verify token can be decoded
        payload = auth.verify_token(token)
        if payload['user_id'] != "test_user" or payload['permissions'] != ["read"]:
            print("[FAIL] Token payload incorrect")
            return False

        print("[PASS] AuthManager uses JWT_SECRET_KEY from environment")
        return True
    finally:
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key
        else:
            os.environ.pop('JWT_SECRET_KEY', None)


def test_tokens_consistent_across_restarts():
    """Test that tokens work across 'restarts' with same secret."""
    from langvel.auth.manager import AuthManager

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
        if payload['user_id'] != "user1" or payload['permissions'] != ["admin"]:
            print("[FAIL] Token not valid across restart simulation")
            return False

        print("[PASS] Tokens work consistently across restarts")
        return True
    finally:
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key
        else:
            os.environ.pop('JWT_SECRET_KEY', None)


def test_tokens_fail_with_different_secret():
    """Test that tokens from different secrets are invalid."""
    from langvel.auth.manager import AuthManager, AuthenticationError

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
        try:
            auth2.verify_token(token)
            print("[FAIL] Token should be invalid with different secret")
            return False
        except AuthenticationError as e:
            if "Invalid token" not in str(e):
                print(f"[FAIL] Wrong error message: {e}")
                return False
            print("[PASS] Tokens correctly fail with different secrets")
            return True
    finally:
        if old_key:
            os.environ['JWT_SECRET_KEY'] = old_key
        else:
            os.environ.pop('JWT_SECRET_KEY', None)


if __name__ == '__main__':
    print("="*60)
    print("Testing JWT Secret Key Validation (TODO-001)")
    print("="*60)
    print()

    tests = [
        ("JWT_SECRET_KEY required check", test_jwt_secret_required),
        ("JWT_SECRET_KEY from environment", test_jwt_secret_from_environment),
        ("Tokens consistent across restarts", test_tokens_consistent_across_restarts),
        ("Tokens fail with different secret", test_tokens_fail_with_different_secret),
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
