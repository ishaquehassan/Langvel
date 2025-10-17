# Changelog

All notable changes to Langvel will be documented in this file.

## [0.2.0] - 2025-01-17

### Major Features Added

#### 🛠️ Tool Execution System (COMPLETE)
- **Full tool registry execution engine** with retry, fallback, and timeout support
- Support for all tool types: custom, RAG, MCP, HTTP, LLM
- Exponential backoff retry logic with configurable attempts (default: 3)
- Tool execution statistics tracking (calls, duration, success rate)
- Comprehensive error handling with `ToolExecutionError`
- Async and sync function support
- Tool performance metrics and monitoring

**Usage:**
```python
@tool(description="Analyze sentiment", retry=5, timeout=30, fallback=fallback_func)
async def analyze_sentiment(self, text: str) -> float:
    return sentiment_score
```

#### 💾 State Persistence (COMPLETE)
- **Full PostgreSQL checkpointer** with asyncpg
  - Thread-based checkpoint management
  - Checkpoint history and versioning
  - Parent-child checkpoint relationships
  - Indexed queries for performance

- **Complete Redis checkpointer** with async support
  - TTL support (24h default, configurable)
  - Thread-based checkpoint lists
  - Fast in-memory state access
  - Automatic cleanup of old checkpoints

**Usage:**
```python
class MyAgent(Agent):
    checkpointer = "postgres"  # or "redis" or "memory"
```

#### 📊 Observability (COMPLETE)
- **LangSmith integration** for tracing
  - Automatic trace and span creation
  - Run tracking with metadata
  - Error logging and debugging

- **Langfuse integration** for observability
  - Token usage tracking
  - Cost monitoring
  - Performance analytics

- **Automatic tracing** in Agent.invoke()
  - LLM call logging with token usage
  - Tool execution tracking
  - Agent lifecycle tracing
  - Error tracking and reporting

**Configuration (.env):**
```bash
LANGSMITH_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
```

#### 🔐 Authentication & Authorization (COMPLETE)
- **JWT token management**
  - Token creation, verification, and refresh
  - Configurable expiry times
  - Secure signing with HS256

- **API key system**
  - Secure key generation
  - Usage tracking and analytics
  - Key revocation support

- **Session management**
  - Activity tracking
  - Session revocation
  - Metadata support

- **RBAC (Role-Based Access Control)**
  - Wildcard permissions (e.g., `admin.*`)
  - Permission inheritance
  - Fine-grained access control

**Usage:**
```python
from langvel.auth.manager import get_auth_manager

auth = get_auth_manager()
token = auth.create_token(user_id="user123", permissions=["read", "write"])
verified = auth.verify_token(token)
```

#### 🤝 Multi-Agent System (COMPLETE)
- **Message bus** for agent communication
  - Pub/sub messaging
  - Direct messaging
  - Broadcast capabilities
  - Priority-based message routing
  - Message history and correlation

- **Agent coordinator**
  - Sequential workflows
  - Parallel execution
  - Conditional routing
  - Shared state management

- **SupervisorAgent pattern**
  - Worker agent coordination
  - Result aggregation
  - Dynamic routing
  - Fault tolerance

**Usage:**
```python
from langvel.multiagent import SupervisorAgent

class Myupervisor(SupervisorAgent):
    def __init__(self):
        super().__init__(workers=[ResearchAgent, AnalysisAgent])

result = await supervisor.invoke({"task": "complex task"})
```

### Enhanced Features

#### 🎨 Tool Decorators
- Added `retry`, `timeout`, and `fallback` parameters to all tool decorators
- Better error messages and validation
- Metadata preservation

#### 🔍 Error Handling
- New `ToolExecutionError` exception class
- Detailed error messages with context
- Automatic error logging to observability platforms
- Graceful degradation support

#### 📦 Dependencies Added
- `asyncpg>=0.29.0` - PostgreSQL async support
- `redis[asyncio]>=5.0.0` - Redis async operations
- `aiohttp>=3.9.0` - HTTP tool execution
- `langsmith>=0.1.0` - LangSmith tracing
- `langfuse>=2.0.0` - Langfuse observability
- `PyJWT>=2.8.0` - JWT token handling
- `langchain-chroma>=0.1.0` - Vector store support

### Configuration Updates

#### New Environment Variables
```bash
# Observability
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=langvel
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Authentication
JWT_SECRET_KEY=
```

### Breaking Changes
None. All changes are additions and enhancements.

### Migration Guide

#### Upgrading from 0.1.0

1. **Update dependencies:**
```bash
pip install -e .
```

2. **Add new environment variables** (optional):
```bash
# For observability
LANGSMITH_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
```

3. **Update checkpointers** (if using postgres/redis):
```python
# Old (stub)
class MyAgent(Agent):
    checkpointer = "postgres"  # Didn't work

# New (fully functional)
class MyAgent(Agent):
    checkpointer = "postgres"  # Works! Creates tables automatically
```

### Performance Improvements
- Async operations throughout for better concurrency
- Connection pooling for PostgreSQL (max 10 connections)
- Redis with automatic TTL cleanup
- Tool execution statistics for monitoring
- Message queue for efficient agent communication

### Documentation Updates
- This CHANGELOG added
- Inline code documentation improved
- Type hints enhanced throughout
- Example usage added to docstrings

### What's Still TODO (for v0.3.0)

#### High Priority
1. **Human-in-the-loop** - State interrupts and approval workflows
2. **Testing framework** - Comprehensive unit and integration tests
3. **Reflection patterns** - Self-correction and validation
4. **Memory systems** - Long-term, semantic, episodic memory

#### Medium Priority
5. **Advanced RAG** - Re-ranking, hybrid search, chunking strategies
6. **MCP enhancements** - Health checks, reconnection, error handling
7. **Rate limiting** - Distributed rate limiting with Redis
8. **WebSocket support** - Bidirectional real-time communication

#### Low Priority
9. **Docker support** - Production-ready containers
10. **CLI enhancements** - More scaffolding commands
11. **Agent templates** - Pre-built agent patterns
12. **Dashboard** - Web UI for monitoring

### Contributors
- Claude Code (AI Assistant)
- Langvel Community

### Links
- [Repository](https://github.com/yourusername/langvel)
- [Documentation](https://langvel.dev)
- [Report Issues](https://github.com/yourusername/langvel/issues)

---

## [0.1.0] - 2025-01-10

### Initial Release
- Basic agent system with LangGraph integration
- State models with Pydantic
- Tool decorators (decorators only, no execution)
- Middleware system (basic)
- CLI tool with generators
- FastAPI server
- Basic RAG and MCP managers (stubs)
- Documentation and examples
