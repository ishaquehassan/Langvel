# Contributing to Langvel

Thank you for your interest in contributing to Langvel! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

### 1. Report Bugs
Found a bug? Help us fix it!

**Before reporting:**
- Check if the bug has already been reported in [Issues](https://github.com/ishaquehassan/langvel/issues)
- Make sure you're using the latest version

**When reporting:**
- Use a clear, descriptive title
- Describe the steps to reproduce
- Include error messages and stack traces
- Mention your Python version and OS

**Template:**
```markdown
**Description:**
Brief description of the bug

**Steps to Reproduce:**
1. Step one
2. Step two
3. ...

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- Langvel version: x.x.x
- Python version: 3.x.x
- OS: macOS/Linux/Windows
- LangGraph version: x.x.x

**Additional Context:**
Any other relevant information
```

### 2. Suggest Features
Have an idea for a new feature?

- Open an issue with the label `enhancement`
- Clearly describe the feature and its use case
- Explain why this feature would be useful
- Consider if it fits Langvel's philosophy

### 3. Improve Documentation
Documentation improvements are always welcome!

- Fix typos or clarify explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation

### 4. Submit Code Changes
Ready to contribute code? Awesome!

## 🚀 Getting Started

### 1. Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/langvel.git
cd langvel

# Add upstream remote
git remote add upstream https://github.com/ishaquehassan/langvel.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### 3. Create a Branch

```bash
# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

## 💻 Development Workflow

### 1. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings to classes and functions
- Include type hints where appropriate

**Code Style:**
```python
# Good example
class MyAgent(Agent):
    """
    Description of what this agent does.

    Example:
        agent = MyAgent()
        result = await agent.invoke({"query": "test"})
    """

    state_model = MyState
    middleware = ['logging']

    async def process(self, state: MyState) -> MyState:
        """
        Process the state.

        Args:
            state: The current state

        Returns:
            The updated state
        """
        # Implementation
        return state
```

### 2. Write Tests

Add tests for your changes:

```bash
# Run tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=langvel
```

**Test Structure:**
```python
import pytest
from langvel.core.agent import Agent

@pytest.mark.asyncio
async def test_your_feature():
    """Test description."""
    agent = MyAgent()
    result = await agent.invoke({"input": "test"})

    assert result["output"] is not None
    assert result["status"] == "success"
```

### 3. Update Documentation

If you added a feature:
- Update relevant documentation files
- Add docstrings to new classes/functions
- Update README.md if needed
- Add examples if appropriate

### 4. Commit Your Changes

Follow conventional commit format:

```bash
# Commit format:
# <type>: <description>
#
# [optional body]
#
# [optional footer]

git add .
git commit -m "feat: Add support for custom checkpointers

- Implement custom checkpointer interface
- Add Redis checkpointer
- Update documentation

Closes #123"
```

**Commit types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Build/tooling changes

### 5. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally (`pytest`)
- [ ] Code follows project style
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How has this been tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Follows code style

## Related Issues
Closes #123
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged!

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Specific Tests

```bash
# Test specific file
pytest tests/test_agent.py

# Test specific function
pytest tests/test_agent.py::test_invoke

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=langvel --cov-report=html
```

### Writing Tests

- Write unit tests for new functions/methods
- Write integration tests for new features
- Use fixtures for common setup
- Mock external dependencies

## 📝 Documentation

Documentation lives in the [langvel-docs](https://github.com/ishaquehassan/langvel-docs) repository.

To contribute documentation:

1. Fork the langvel-docs repository
2. Make your changes
3. Submit a pull request

Documentation uses VitePress and is written in Markdown.

## 🎨 Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small
- Use meaningful variable names

### Example

```python
from typing import Dict, Any, Optional
from langvel.core.agent import Agent
from langvel.state.base import StateModel

class MyState(StateModel):
    """State model for MyAgent."""

    query: str
    response: str = ""

class MyAgent(Agent):
    """
    Agent that processes queries.

    This agent takes a query and generates a response using LLM.

    Attributes:
        state_model: The state model class
        middleware: List of middleware to apply

    Example:
        >>> agent = MyAgent()
        >>> result = await agent.invoke({"query": "Hello"})
        >>> print(result["response"])
    """

    state_model = MyState
    middleware = ['logging']

    def build_graph(self):
        """Build the agent's execution graph."""
        return self.start().then(self.process).end()

    async def process(self, state: MyState) -> MyState:
        """
        Process the query and generate response.

        Args:
            state: Current state with query

        Returns:
            Updated state with response
        """
        response = await self.llm.invoke(state.query)
        state.response = response
        return state
```

## 🐛 Reporting Security Issues

**DO NOT** report security vulnerabilities in public issues.

Instead, email: [your-email@example.com]

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## 💬 Community

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Trolling or inflammatory comments
- Publishing private information

## ❓ Questions?

- Check [existing issues](https://github.com/ishaquehassan/langvel/issues)
- Read the [documentation](https://ishaquehassan.github.io/langvel-docs/)
- Ask in [GitHub Discussions](https://github.com/ishaquehassan/langvel/discussions)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Langvel! 🎉

Your contributions help make Langvel better for everyone.
