#!/usr/bin/env python3
"""
Langvel Framework Setup Script

Quick setup script for developers who want to get started immediately.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header():
    """Print welcome header."""
    print("=" * 60)
    print("🚀 Langvel Framework Setup")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("✓ Checking Python version...")
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor} detected\n")


def create_venv():
    """Create virtual environment."""
    print("📦 Creating virtual environment...")
    venv_path = Path("venv")

    if venv_path.exists():
        print("  ⚠  Virtual environment already exists\n")
        return True

    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("  ✓ Virtual environment created\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to create virtual environment: {e}\n")
        return False


def get_venv_python():
    """Get path to Python in virtual environment."""
    if sys.platform == "win32":
        return Path("venv") / "Scripts" / "python.exe"
    else:
        return Path("venv") / "bin" / "python"


def get_venv_pip():
    """Get path to pip in virtual environment."""
    if sys.platform == "win32":
        return Path("venv") / "Scripts" / "pip.exe"
    else:
        return Path("venv") / "bin" / "pip"


def upgrade_pip():
    """Upgrade pip in virtual environment."""
    print("⬆️  Upgrading pip...")
    python_path = get_venv_python()

    try:
        subprocess.run(
            [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("  ✓ Pip upgraded\n")
        return True
    except subprocess.CalledProcessError:
        print("  ⚠  Pip upgrade warning (continuing anyway)\n")
        return True


def install_dependencies():
    """Install Langvel and dependencies."""
    print("📚 Installing dependencies...")
    print("  This may take a few minutes...\n")
    pip_path = get_venv_pip()

    try:
        # Install in editable mode
        subprocess.run(
            [str(pip_path), "install", "-e", "."],
            check=True
        )
        print("\n  ✓ Dependencies installed\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ Failed to install dependencies: {e}\n")
        return False


def initialize_project():
    """Initialize project structure."""
    print("🏗️  Initializing project structure...")

    # Create directories
    dirs = [
        'app/agents',
        'app/middleware',
        'app/tools',
        'app/models',
        'storage/logs',
        'storage/checkpoints',
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create __init__ files
    for dir_path in ['app', 'app/agents', 'app/middleware', 'app/tools', 'app/models']:
        init_file = Path(dir_path) / '__init__.py'
        init_file.touch(exist_ok=True)

    print("  ✓ Project structure created\n")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("✨ Setup Complete!")
    print("=" * 60)
    print()
    print("To activate the virtual environment:")

    if sys.platform == "win32":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")

    print()
    print("Next steps:")
    print("  1. Update .env with your API keys")
    print("  2. Create your first agent:")
    print("     langvel make:agent MyAgent")
    print("  3. Register it in routes/agent.py")
    print("  4. Start the server:")
    print("     langvel agent serve")
    print()
    print("For help:")
    print("  langvel --help")
    print()


def main():
    """Main setup function."""
    print_header()

    # Check Python version
    check_python_version()

    # Create virtual environment
    if not create_venv():
        sys.exit(1)

    # Upgrade pip
    if not upgrade_pip():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Initialize project
    if not initialize_project():
        sys.exit(1)

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
