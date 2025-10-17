# -*- coding: utf-8 -*-
"""Test agent path validation (TODO-004)."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_validation_function_exists():
    """Test that validation function exists."""
    try:
        from langvel.server import validate_agent_path, AGENT_PATH_PATTERN

        assert validate_agent_path is not None
        assert AGENT_PATH_PATTERN is not None

        print("[PASS] Validation function and pattern exist")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_valid_agent_paths():
    """Test that valid agent paths pass validation."""
    from langvel.server import validate_agent_path

    valid_paths = [
        "my-agent",
        "agents/research",
        "api/v1/analyzer",
        "test_agent_123",
        "data-processing/transformer",
        "ml-models/bert-classifier",
    ]

    try:
        for path in valid_paths:
            # Should not raise exception
            validate_agent_path(path)

        print(f"[PASS] All {len(valid_paths)} valid paths accepted")
        return True
    except Exception as e:
        print(f"[FAIL] Valid path rejected: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_path_traversal_blocked():
    """Test that path traversal attempts are blocked."""
    from langvel.server import validate_agent_path
    from fastapi.exceptions import HTTPException

    malicious_paths = [
        "../etc/passwd",
        "../../secret",
        "agents/../../../etc/hosts",
        "valid/../../invalid",
        "..hidden",
    ]

    for path in malicious_paths:
        try:
            validate_agent_path(path)
            print(f"[FAIL] Path traversal not blocked: {path}")
            return False
        except HTTPException as e:
            if e.status_code != 400:
                print(f"[FAIL] Wrong status code for {path}: {e.status_code}")
                return False
            if "traversal" not in e.detail.lower():
                print(f"[FAIL] Wrong error message for {path}: {e.detail}")
                return False

    print(f"[PASS] All {len(malicious_paths)} path traversal attempts blocked")
    return True


def test_absolute_paths_blocked():
    """Test that absolute paths are blocked."""
    from langvel.server import validate_agent_path
    from fastapi.exceptions import HTTPException

    absolute_paths = [
        "/etc/passwd",
        "/agents/research",
        "/var/log/app.log",
    ]

    for path in absolute_paths:
        try:
            validate_agent_path(path)
            print(f"[FAIL] Absolute path not blocked: {path}")
            return False
        except HTTPException as e:
            if e.status_code != 400:
                print(f"[FAIL] Wrong status code for {path}: {e.status_code}")
                return False
            if "relative" not in e.detail.lower():
                print(f"[FAIL] Wrong error message for {path}: {e.detail}")
                return False

    print(f"[PASS] All {len(absolute_paths)} absolute paths blocked")
    return True


def test_special_characters_blocked():
    """Test that special characters are blocked."""
    from langvel.server import validate_agent_path
    from fastapi.exceptions import HTTPException

    invalid_paths = [
        "agent;rm -rf /",
        "agent && whoami",
        "agent|cat /etc/passwd",
        "agent`id`",
        "agent$(ls)",
        "agent<script>alert(1)</script>",
        "agent%20with%20spaces",
        "agent\nwith\nnewlines",
        "agent\x00null",
    ]

    for path in invalid_paths:
        try:
            validate_agent_path(path)
            print(f"[FAIL] Invalid characters not blocked: {path}")
            return False
        except HTTPException as e:
            if e.status_code != 400:
                print(f"[FAIL] Wrong status code for {path}: {e.status_code}")
                return False

    print(f"[PASS] All {len(invalid_paths)} paths with special characters blocked")
    return True


def test_consecutive_slashes_blocked():
    """Test that consecutive slashes are blocked."""
    from langvel.server import validate_agent_path
    from fastapi.exceptions import HTTPException

    # Note: paths starting with // are caught by absolute path check first
    paths_with_double_slashes = [
        "agents//research",
        "api//v1//analyzer",
        "middle//slashes//here",
    ]

    for path in paths_with_double_slashes:
        try:
            validate_agent_path(path)
            print(f"[FAIL] Consecutive slashes not blocked: {path}")
            return False
        except HTTPException as e:
            if e.status_code != 400:
                print(f"[FAIL] Wrong status code for {path}: {e.status_code}")
                return False
            # Should mention either consecutive slashes or invalid characters
            if "consecutive" not in e.detail.lower() and "invalid" not in e.detail.lower():
                print(f"[FAIL] Unexpected error message for {path}: {e.detail}")
                return False

    print(f"[PASS] All {len(paths_with_double_slashes)} paths with consecutive slashes blocked")
    return True


def test_empty_path_blocked():
    """Test that empty paths are blocked."""
    from langvel.server import validate_agent_path
    from fastapi.exceptions import HTTPException

    empty_paths = ["", "   ", "\t", "\n"]

    for path in empty_paths:
        try:
            validate_agent_path(path)
            print(f"[FAIL] Empty path not blocked: '{path}'")
            return False
        except HTTPException as e:
            if e.status_code != 400:
                print(f"[FAIL] Wrong status code for empty path: {e.status_code}")
                return False
            if "empty" not in e.detail.lower():
                print(f"[FAIL] Wrong error message for empty path: {e.detail}")
                return False

    print(f"[PASS] All {len(empty_paths)} empty paths blocked")
    return True


def test_pattern_validation():
    """Test the regex pattern directly."""
    from langvel.server import AGENT_PATH_PATTERN

    # Valid patterns
    valid = ["agent", "my-agent", "api/v1/test", "test_123", "a-b-c/d-e-f"]
    for path in valid:
        if not AGENT_PATH_PATTERN.match(path):
            print(f"[FAIL] Pattern rejected valid path: {path}")
            return False

    # Invalid patterns
    invalid = ["agent!", "test@domain", "path with spaces", "agent;cmd", "a&b"]
    for path in invalid:
        if AGENT_PATH_PATTERN.match(path):
            print(f"[FAIL] Pattern accepted invalid path: {path}")
            return False

    print("[PASS] Regex pattern validation correct")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Testing Agent Path Validation (TODO-004)")
    print("="*60)
    print()

    tests = [
        ("Validation function exists", test_validation_function_exists),
        ("Valid agent paths accepted", test_valid_agent_paths),
        ("Path traversal blocked", test_path_traversal_blocked),
        ("Absolute paths blocked", test_absolute_paths_blocked),
        ("Special characters blocked", test_special_characters_blocked),
        ("Consecutive slashes blocked", test_consecutive_slashes_blocked),
        ("Empty paths blocked", test_empty_path_blocked),
        ("Regex pattern validation", test_pattern_validation),
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
