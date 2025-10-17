# Langvel Installation Guide

Complete guide to installing and setting up Langvel framework.

## Prerequisites

- **Python 3.10+** (required)
- **pip** (usually comes with Python)
- **Git** (for cloning the repository)

## Installation Methods

### Method 1: Quick Setup (Recommended)

The fastest way to get started with Langvel:

```bash
# Clone the repository
git clone https://github.com/yourusername/langvel.git
cd langvel

# Run the setup script
python setup.py
```

This will:
1. Create a virtual environment
2. Upgrade pip
3. Install all dependencies
4. Initialize project structure

**Then activate the virtual environment:**

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Method 2: Using CLI Setup Command

If you already have Langvel installed globally or want more control:

```bash
# Install Langvel first
pip install -e .

# Run setup with venv
langvel setup --with-venv
```

This provides a beautiful progress interface with Rich output!

### Method 3: Manual Installation

For developers who prefer manual control:

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install Langvel
pip install -e .

# 5. Initialize project
langvel init
```

## Post-Installation

### 1. Configure Environment Variables

Edit `.env` file with your API keys:

```bash
# Required for most features
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx

# Optional - for specific features
SLACK_BOT_TOKEN=xoxb-xxxxx
GITHUB_TOKEN=ghp_xxxxx
```

### 2. Verify Installation

```bash
# Check Langvel version
langvel --version

# List available commands
langvel --help

# Verify CLI is working
langvel agent list
```

### 3. Create Your First Agent

```bash
# Generate agent
langvel make:agent MyFirstAgent

# Generate state model
langvel make:state MyFirstAgentState

# View the created files
ls -la app/agents/
```

### 4. Test the Setup

Create a simple test agent:

```python
# app/agents/test_agent.py
from langvel.core.agent import Agent
from langvel.state.base import StateModel

class TestState(StateModel):
    query: str
    response: str = ""

class TestAgent(Agent):
    state_model = TestState

    def build_graph(self):
        return self.start().then(self.handle).end()

    async def handle(self, state: TestState):
        state.response = f"Processed: {state.query}"
        return state
```

Register it in `routes/agent.py`:

```python
from langvel.routing.router import router
from app.agents.test_agent import TestAgent

@router.flow('/test')
class TestFlow(TestAgent):
    pass
```

Test it:

```bash
langvel agent test /test -i '{"query": "Hello Langvel!"}'
```

### 5. Start the Server

```bash
# Development server with auto-reload
langvel agent serve --reload

# Production server
langvel agent serve --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for API documentation.

## Troubleshooting

### Python Version Issues

**Error:** `Python 3.10 or higher is required`

**Solution:**
```bash
# Check your Python version
python --version

# Install Python 3.10+ from python.org or use pyenv
pyenv install 3.11
pyenv local 3.11
```

### Virtual Environment Issues

**Error:** `venv module not found`

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-venv

# macOS (via Homebrew)
brew install python@3.11

# Or use virtualenv
pip install virtualenv
virtualenv venv
```

### Permission Issues (Linux/macOS)

**Error:** `Permission denied`

**Solution:**
```bash
# Make setup script executable
chmod +x setup.py

# Or run with python
python setup.py
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'langvel'`

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process or use different port
langvel agent serve --port 8080
```

## Development Installation

For contributing to Langvel:

```bash
# Clone repository
git clone https://github.com/yourusername/langvel.git
cd langvel

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if available)
pre-commit install

# Run tests
pytest

# Run linting
black .
ruff check .
```

## Updating Langvel

To update to the latest version:

```bash
# Activate virtual environment
source venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies
pip install -e . --upgrade
```

## Uninstallation

To remove Langvel:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove Langvel files
cd ..
rm -rf langvel
```

## Docker Installation (Optional)

Coming soon! We're working on official Docker images.

## System Requirements

### Minimum
- Python 3.10+
- 2GB RAM
- 1GB disk space

### Recommended
- Python 3.11+
- 4GB RAM
- 2GB disk space
- SSD for better performance

## Platform Support

Langvel is tested and supported on:

- ✅ macOS (Intel and Apple Silicon)
- ✅ Linux (Ubuntu 20.04+, Debian, CentOS)
- ✅ Windows 10/11 (via WSL2 recommended)

## Getting Help

- 📚 [Documentation](https://langvel.dev)
- 💬 [Discord Community](https://discord.gg/langvel)
- 🐛 [Issue Tracker](https://github.com/yourusername/langvel/issues)
- 📧 [Email Support](mailto:support@langvel.dev)

## Next Steps

Once installed, check out:

1. [Quick Start Guide](./README.md#quick-start)
2. [Examples Directory](./examples/)
3. [API Documentation](https://langvel.dev/api)
4. [Tutorial Series](https://langvel.dev/tutorials)

Happy building with Langvel! 🚀
