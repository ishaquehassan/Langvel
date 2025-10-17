# Langvel 🚀

**A Laravel-inspired framework for building LangGraph agents with elegant developer experience.**

Langvel brings the beloved Laravel development experience to AI agent development, making it easy to build powerful, production-ready agents using LangGraph under the hood.

## ✨ Features

- **🎯 Laravel-Inspired DX**: Familiar routing, middleware, and service provider patterns
- **🔧 Full LangGraph Power**: Access all LangGraph capabilities with elegant abstractions
- **🤖 Built-in LLM Support**: Every agent has `self.llm` ready to use (Claude, GPT)
- **🧠 RAG Integration**: Built-in support for vector stores and embeddings
- **🔌 MCP Servers**: Seamless Model Context Protocol integration
- **🛠️ Tool System**: Full execution engine with retry, fallback, and timeout
- **🔐 Authentication & Authorization**: JWT, API keys, RBAC, and session management
- **🎨 Request/Response Modeling**: Pydantic-based state management
- **⚡ CLI Tool**: Artisan-style commands for scaffolding and management
- **🌊 Streaming Support**: Built-in streaming for real-time responses
- **💾 Checkpointers**: Production-ready PostgreSQL and Redis persistence
- **📊 Observability**: LangSmith and Langfuse integration for tracing
- **📝 Structured Logging**: JSON logging with full context for ELK, Datadog, CloudWatch
- **🤝 Multi-Agent**: Agent coordination, message bus, and supervisor patterns

## 🚀 Quick Start

### One-Command Setup (Recommended)

The fastest way to get started:

```bash
# Clone and navigate
git clone https://github.com/yourusername/langvel.git
cd langvel

# Run automated setup (creates venv, installs dependencies, initializes project)
python setup.py

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Configure your API keys
nano .env  # Add your ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### Alternative Installation Methods

**Using CLI Setup Command:**
```bash
pip install -e .
langvel setup --with-venv  # Beautiful progress interface!
```

**Manual Installation:**
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
langvel init
```

📚 For detailed installation instructions, see [INSTALL.md](./INSTALL.md)

### Create Your First Agent

```bash
# Generate a new agent
langvel make:agent CustomerSupport

# Generate a state model
langvel make:state CustomerSupportState

# Generate middleware
langvel make:middleware RateLimit

# Generate a tool
langvel make:tool SentimentAnalyzer
```

### Define Your Agent

```python
# app/agents/customer_support.py
from langvel.core.agent import Agent
from langvel.state.base import StateModel
from langvel.tools.decorators import tool, rag_tool
from pydantic import Field

class CustomerSupportState(StateModel):
    query: str
    response: str = ""
    category: str = "general"

class CustomerSupportAgent(Agent):
    state_model = CustomerSupportState
    middleware = ['logging', 'rate_limit']

    def build_graph(self):
        return (
            self.start()
            .then(self.classify)
            .then(self.search_knowledge)
            .then(self.generate_response)
            .end()
        )

    async def classify(self, state: CustomerSupportState):
        # Your classification logic
        return state

    @rag_tool(collection='knowledge_base', k=5)
    async def search_knowledge(self, state: CustomerSupportState):
        # RAG retrieval happens automatically
        return state

    async def generate_response(self, state: CustomerSupportState):
        # Generate response using LLM
        return state
```

### Register Routes

```python
# routes/agent.py
from langvel.routing.router import router
from app.agents.customer_support import CustomerSupportAgent

@router.flow('/customer-support', middleware=['auth', 'rate_limit'])
class CustomerSupportFlow(CustomerSupportAgent):
    pass
```

### Run Your Agent

```bash
# Start the server
langvel agent serve --port 8000

# Or run directly
python -c "
import asyncio
from app.agents.customer_support import CustomerSupportAgent

async def main():
    agent = CustomerSupportAgent()
    result = await agent.invoke({
        'query': 'How do I reset my password?'
    })
    print(result)

asyncio.run(main())
"
```

## 📚 Core Concepts

### 1. Agents (Controllers)

Agents are like Laravel controllers - they define the logic and workflow.

```python
class MyAgent(Agent):
    state_model = MyState
    middleware = ['logging', 'auth']

    def build_graph(self):
        return self.start().then(self.handle).end()

    async def handle(self, state):
        # Your logic here
        return state
```

### 2. State Models (Eloquent-like)

State models use Pydantic for validation and type safety.

```python
class MyState(StateModel):
    user_id: str
    query: str
    response: Optional[str] = None

    class Config:
        checkpointer = 'postgres'
        interrupts = ['before_response']
```

### 3. Routing

Define agent routes with a familiar syntax.

```python
router = AgentRouter()

@router.flow('/my-agent', middleware=['auth'])
class MyAgentFlow(MyAgent):
    pass

# Route groups
with router.group(prefix='/admin', middleware=['admin']):
    @router.flow('/dashboard')
    class AdminDashboard(Agent):
        pass
```

### 4. Middleware

Add cross-cutting concerns like authentication, logging, and rate limiting.

```python
class MyMiddleware(Middleware):
    async def before(self, state):
        # Run before agent
        return state

    async def after(self, state):
        # Run after agent
        return state
```

### 5. Tools

Define tools with elegant decorators.

```python
# Custom tool
@tool(description="Process data")
async def process_data(self, data: str) -> str:
    return processed_data

# RAG tool
@rag_tool(collection='docs', k=5)
async def search_docs(self, query: str):
    pass  # Returns retrieved documents

# MCP tool
@mcp_tool(server='slack', tool_name='send_message')
async def send_slack(self, message: str):
    pass  # Calls MCP server

# HTTP tool
@http_tool(method='POST', url='https://api.example.com')
async def call_api(self, params: dict):
    pass  # Makes HTTP request

# LLM tool
@llm_tool(system_prompt="You are a code reviewer")
async def review_code(self, code: str) -> str:
    pass  # Calls LLM
```

### 6. RAG Integration

Built-in support for vector stores and RAG.

```python
# Configure in config/langvel.py
RAG_PROVIDER = 'chroma'
RAG_EMBEDDING_MODEL = 'openai/text-embedding-3-small'

# Use in agents
@rag_tool(collection='knowledge_base', k=5)
async def search_knowledge(self, state):
    # Retrieved docs added to state automatically
    return state
```

### 7. MCP Servers

Integrate external tools via Model Context Protocol.

```python
# Configure in config/langvel.py
MCP_SERVERS = {
    'slack': {
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-slack'],
        'env': {'SLACK_BOT_TOKEN': os.getenv('SLACK_BOT_TOKEN')}
    }
}

# Use in agents
@mcp_tool(server='slack', tool_name='send_message')
async def notify_slack(self, message: str):
    pass
```

### 8. LLM Integration

Every agent has `self.llm` ready to use - no setup needed!

```python
class MyAgent(Agent):
    async def process(self, state):
        # Simple LLM query
        response = await self.llm.invoke(
            prompt="Explain Python",
            system_prompt="You are a helpful teacher"
        )

        # Streaming response
        async for chunk in self.llm.stream("Tell me a story"):
            print(chunk, end="")

        # Structured output with Pydantic
        from pydantic import BaseModel

        class Analysis(BaseModel):
            sentiment: str
            confidence: float

        llm_structured = self.llm.with_structured_output(Analysis)
        result = await llm_structured.ainvoke("Analyze: ...")

        # Multi-turn conversation
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is..."},
            {"role": "user", "content": "Show me an example"}
        ]
        response = await self.llm.chat(messages)

        return state
```

**Supported Providers:**
- **Anthropic**: Claude 3.5 Sonnet, Opus, Haiku
- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5

**Configuration** in `config/langvel.py`:
```python
LLM_PROVIDER = 'anthropic'  # or 'openai'
LLM_MODEL = 'claude-3-5-sonnet-20241022'
LLM_TEMPERATURE = 0.7
```

### 9. Authentication & Authorization

Complete auth system with JWT, API keys, and RBAC.

```python
from langvel.auth.manager import get_auth_manager
from langvel.auth.decorators import requires_auth, requires_permission, rate_limit

# Create and verify JWT tokens
auth = get_auth_manager()
token = auth.create_token(user_id="user123", permissions=["read", "write"])
verified = auth.verify_token(token)

# API key management
api_key = auth.create_api_key(name="Production API", permissions=["api.*"])
key_data = auth.verify_api_key(api_key)

# Use decorators in agents
@requires_auth
async def sensitive_operation(self, state):
    # Only authenticated users
    pass

@requires_permission('admin')
async def admin_operation(self, state):
    # Only users with 'admin' permission
    pass

@rate_limit(max_requests=5, window=60)
async def expensive_operation(self, state):
    # Rate limited to 5 requests per minute
    pass
```

### 10. Observability & Tracing

Automatic tracing with LangSmith and Langfuse, plus production-ready structured logging.

```python
# Configure in .env
LANGSMITH_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key

# Tracing is automatic for all agents!
# Every agent invocation is traced with:
# - Input/output data
# - LLM calls with token usage
# - Tool executions
# - Error tracking
# - Performance metrics

# Structured JSON logging (production-ready)
from langvel.logging import get_logger, setup_logging

setup_logging(log_level="INFO", log_file="langvel.log", json_format=True)
logger = get_logger(__name__)
logger.info("Agent started", extra={"agent": "CustomerSupport", "user_id": "123"})
```

**JSON Log Output** (compatible with ELK, Datadog, CloudWatch):
```json
{
  "timestamp": "2025-10-17T11:13:25.502449Z",
  "level": "INFO",
  "logger": "langvel.middleware.logging",
  "message": "Agent execution started",
  "event": "agent_input",
  "state_keys": ["query", "user_id"]
}
```

### 11. Multi-Agent Systems

Coordinate multiple agents working together.

```python
from langvel.multiagent import SupervisorAgent, AgentCoordinator

# Define worker agents
class ResearchAgent(Agent):
    def build_graph(self):
        return self.start().then(self.research).end()

class AnalysisAgent(Agent):
    def build_graph(self):
        return self.start().then(self.analyze).end()

# Create supervisor that coordinates workers
class TaskSupervisor(SupervisorAgent):
    def __init__(self):
        super().__init__(workers=[ResearchAgent, AnalysisAgent])

# Execute with automatic coordination
supervisor = TaskSupervisor()
result = await supervisor.invoke({"task": "Complex research task"})
```

## 🛠️ CLI Commands

### Setup & Installation

```bash
langvel setup                   # Initialize project structure
langvel setup --with-venv       # Setup with virtual environment and dependencies
python setup.py                 # Alternative automated setup script
```

### Project Management

```bash
langvel init                    # Initialize new project
langvel agent serve             # Start development server
langvel agent serve --reload    # Start with auto-reload
```

### Generators

```bash
langvel make:agent MyAgent              # Create agent
langvel make:state MyState              # Create state model
langvel make:middleware MyMiddleware    # Create middleware
langvel make:tool MyTool               # Create tool
```

### Agent Management

```bash
langvel agent list                      # List all agents
langvel agent test /my-agent -i '{"query":"test"}'  # Test agent
langvel agent graph /my-agent -o graph.png          # Visualize graph
```

## 📖 Example Agents

Check out `app/agents/` for working example implementations:

- **customer_support_agent.py**: Complete example showing RAG, MCP, sentiment analysis, conditional routing, and middleware
- **code_review_agent.py**: LLM integration example with direct invocation, streaming, structured output, and multi-turn conversations

Both examples are production-ready and demonstrate best practices!

## 🏗️ Architecture

```
langvel/
├── core/              # Core agent and graph builder
├── routing/           # Router and route management
├── state/             # State models and checkpointers
├── tools/             # Tool decorators and registry
├── middleware/        # Middleware system
├── rag/               # RAG manager and config
├── mcp/               # MCP server integration
├── llm/               # LLM manager (Anthropic, OpenAI)
├── auth/              # Authentication decorators
├── cli/               # CLI commands
└── server.py          # FastAPI server

app/                   # Your application (like Laravel's app/)
├── agents/            # Agent classes (like Controllers)
│   ├── customer_support_agent.py
│   └── code_review_agent.py
├── middleware/        # Custom middleware
├── tools/             # Custom tools
├── models/            # State models (like Eloquent models)
└── providers/         # Service providers

config/                # Configuration files
routes/                # Route definitions
docs/                  # Additional documentation
```

## 🔧 Configuration

All configuration is in `config/langvel.py`:

```python
# LLM
LLM_PROVIDER = 'anthropic'
LLM_MODEL = 'claude-3-5-sonnet-20241022'

# RAG
RAG_PROVIDER = 'chroma'
RAG_EMBEDDING_MODEL = 'openai/text-embedding-3-small'

# MCP Servers
MCP_SERVERS = {
    'slack': {...},
    'github': {...}
}

# State
STATE_CHECKPOINTER = 'memory'  # or 'postgres', 'redis'

# Server
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8000
```

## 🌐 HTTP API

When running the server, agents are accessible via HTTP:

```bash
# List agents
GET /agents

# Invoke agent
POST /agents/my-agent
{
  "input": {"query": "hello"},
  "stream": false
}

# Stream agent
POST /agents/my-agent
{
  "input": {"query": "hello"},
  "stream": true
}

# Get graph visualization
GET /agents/my-agent/graph
```

## 📚 Additional Documentation

### Detailed Guides

- **[INSTALL.md](./INSTALL.md)** - Complete installation guide with troubleshooting
- **[QUICKSTART.md](./QUICKSTART.md)** - 5-minute tutorial to get started fast
- **[FEATURES.md](./FEATURES.md)** - Complete feature list with examples
- **[docs/LLM_GUIDE.md](./docs/LLM_GUIDE.md)** - Comprehensive LLM integration guide
- **[LARAVEL_COMPARISON.md](./LARAVEL_COMPARISON.md)** - Laravel patterns mapped to Langvel

### Quick References

**Installation:**
```bash
python setup.py              # One-command setup
source venv/bin/activate     # Activate
langvel make:agent MyAgent   # Create agent
```

**LLM Usage:**
```python
# Every agent has self.llm
response = await self.llm.invoke("Your prompt")
async for chunk in self.llm.stream("Prompt"): ...
result = await self.llm.with_structured_output(MyModel).ainvoke("Prompt")
```

**Common Patterns:**
```python
# Agent with RAG + LLM
class MyAgent(Agent):
    @rag_tool(collection='docs', k=5)
    async def search(self, state):
        return state  # Docs auto-added

    async def respond(self, state):
        response = await self.llm.invoke(
            f"Answer based on: {state.rag_context}",
            system_prompt="You are helpful"
        )
        return state
```

## 🎯 Laravel Developers

If you know Laravel, you already know Langvel!

| Laravel | Langvel |
|---------|---------|
| `Controller` | `Agent` |
| `Model` | `StateModel` |
| `Route::get()` | `@router.flow()` |
| `Middleware` | `Middleware` |
| `php artisan` | `langvel` |
| `config/*.php` | `config/*.py` |

See [LARAVEL_COMPARISON.md](./LARAVEL_COMPARISON.md) for detailed mapping.

## 💡 Best Practices

### 1. Use State Models for Type Safety
```python
class MyState(StateModel):
    query: str
    response: str = ""
    # Pydantic validation automatic!
```

### 2. Leverage self.llm for AI Operations
```python
# Built-in, no setup needed
response = await self.llm.invoke("Your prompt")
```

### 3. Use Decorators for Clean Code
```python
@rag_tool(collection='docs')
@requires_auth
@rate_limit(10, 60)
async def my_node(self, state):
    pass
```

### 4. Test Agents Before Deployment
```bash
langvel agent test /my-agent -i '{"query":"test"}'
```

### 5. Visualize Workflows
```bash
langvel agent graph /my-agent -o graph.png
```

## 🚀 Production Deployment

### Using Docker (Coming Soon)
```dockerfile
FROM python:3.11
COPY . /app
RUN pip install -e .
CMD ["langvel", "agent", "serve"]
```

### Using Systemd
```bash
# Create service file
sudo nano /etc/systemd/system/langvel.service

[Unit]
Description=Langvel Agent Server
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/langvel
ExecStart=/path/to/venv/bin/langvel agent serve
Restart=always

[Install]
WantedBy=multi-user.target
```

### Environment Variables
```bash
# Production .env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
STATE_CHECKPOINTER=postgres
DATABASE_URL=postgresql://...
DEBUG=false
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Laravel**: For the amazing DX that inspired this framework
- **LangGraph**: For the powerful agent orchestration capabilities
- **LangChain**: For the comprehensive LLM tooling
- **Anthropic**: For Claude AI models
- **OpenAI**: For GPT models

## 🔗 Links

- [Documentation](https://langvel.dev)
- [Example Agents](./app/agents)
- [GitHub](https://github.com/yourusername/langvel)
- [Discord](https://discord.gg/langvel)
- [Installation Guide](./INSTALL.md)
- [Quick Start](./QUICKSTART.md)
- [LLM Guide](./docs/LLM_GUIDE.md)

## 🌟 Why Langvel?

**Laravel's Elegance + LangGraph's Power + Built-in LLM**

- ✅ **Familiar Patterns** - If you know Laravel, you know Langvel
- ✅ **Production Ready** - Type-safe, tested, documented
- ✅ **Full-Featured** - RAG, MCP, LLM, Auth, Middleware, Tools
- ✅ **Developer Joy** - Beautiful CLI, one-command setup, great DX
- ✅ **Extensible** - Easy to add custom tools, middleware, providers

---

**Start building amazing AI agents with Laravel-like elegance!** 🚀

Built with ❤️ by the Langvel community
