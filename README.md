# Langvel 🚀

**A Laravel-inspired framework for building LangGraph agents with elegant developer experience.**

Langvel brings the beloved Laravel development experience to AI agent development, making it easy to build powerful, production-ready agents using LangGraph under the hood.

## ✨ Features

- **🎯 Laravel-Inspired DX**: Familiar routing, middleware, and service provider patterns
- **🔧 Full LangGraph Power**: Access all LangGraph capabilities with elegant abstractions
- **🧠 RAG Integration**: Built-in support for vector stores and embeddings
- **🔌 MCP Servers**: Seamless Model Context Protocol integration
- **🛠️ Tool System**: Decorators for custom, RAG, MCP, HTTP, and LLM tools
- **🔐 Authentication & Authorization**: Built-in auth states and permission decorators
- **🎨 Request/Response Modeling**: Pydantic-based state management
- **⚡ CLI Tool**: Artisan-style commands for scaffolding and management
- **🌊 Streaming Support**: Built-in streaming for real-time responses
- **💾 Checkpointers**: Memory, PostgreSQL, and Redis state persistence

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

### 8. Authentication & Authorization

Built-in auth decorators and state management.

```python
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

## 📖 Examples

Check out the `examples/` directory for complete examples:

- **Customer Support Agent**: RAG, MCP, sentiment analysis, routing
- **Code Review Agent**: LLM tools, GitHub integration
- **Data Analysis Agent**: HTTP tools, streaming responses

## 🏗️ Architecture

```
langvel/
├── core/              # Core agent and graph builder
├── routing/           # Router and route management
├── state/             # State models and checkpointers
├── tools/             # Tool decorators and registry
├── middleware/        # Middleware system
├── rag/              # RAG manager and config
├── mcp/              # MCP server integration
├── auth/             # Authentication decorators
├── cli/              # CLI commands
└── server.py         # FastAPI server

app/                  # Your application
├── agents/           # Your agents
├── middleware/       # Custom middleware
├── tools/            # Custom tools
└── models/           # State models

config/               # Configuration
routes/               # Route definitions
examples/             # Example agents
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

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Laravel**: For the amazing DX that inspired this framework
- **LangGraph**: For the powerful agent orchestration capabilities
- **LangChain**: For the comprehensive LLM tooling

## 🔗 Links

- [Documentation](https://langvel.dev)
- [Examples](./examples)
- [GitHub](https://github.com/yourusername/langvel)
- [Discord](https://discord.gg/langvel)

---

Built with ❤️ by the Langvel community
