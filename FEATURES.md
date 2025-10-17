# Langvel Features Overview

Complete feature list and capabilities of the Langvel framework.

## 🎯 Core Features

### 1. Agent System
- **Base Agent Class**: Laravel-inspired controller pattern for agents
- **Graph Builder**: Fluent, chainable API for defining workflows
- **State Management**: Pydantic-based state models with validation
- **Compilation**: Automatic LangGraph compilation from builder patterns

**Example:**
```python
class MyAgent(Agent):
    def build_graph(self):
        return (
            self.start()
            .then(self.step1)
            .branch({'a': self.path_a, 'b': self.path_b})
            .merge(self.final)
            .end()
        )
```

### 2. Routing System
- **Declarative Routes**: `@router.flow()` decorator syntax
- **Route Groups**: Shared prefixes and middleware
- **Metadata Support**: Custom metadata per route
- **Dynamic Discovery**: Automatic route registration

**Example:**
```python
@router.flow('/customer-support', middleware=['auth', 'rate_limit'])
class CustomerSupportFlow(Agent):
    pass
```

### 3. State Models
- **Pydantic Integration**: Full type safety and validation
- **Pre-built Models**: StateModel, AuthenticatedState, RAGState
- **Checkpointing**: Memory, PostgreSQL, Redis support
- **State Interrupts**: Human-in-the-loop control points

**Example:**
```python
class MyState(StateModel):
    query: str
    response: Optional[str] = None

    class Config:
        checkpointer = 'postgres'
        interrupts = ['before_response']
```

### 4. Tool System
- **Custom Tools**: `@tool` decorator
- **RAG Tools**: `@rag_tool` for retrieval
- **MCP Tools**: `@mcp_tool` for external services
- **HTTP Tools**: `@http_tool` for API calls
- **LLM Tools**: `@llm_tool` for LLM operations
- **Tool Registry**: Automatic discovery and management

**Example:**
```python
@tool(description="Analyze sentiment")
async def analyze_sentiment(self, text: str) -> float:
    return sentiment_score

@rag_tool(collection='docs', k=5)
async def search_docs(self, query: str):
    pass  # Auto-retrieval
```

### 5. Middleware System
- **Before/After Hooks**: Intercept agent execution
- **Built-in Middleware**: Logging, auth, rate limiting, CORS, validation
- **Custom Middleware**: Easy to create
- **Middleware Manager**: Automatic execution pipeline
- **Per-Route Middleware**: Fine-grained control

**Example:**
```python
class MyMiddleware(Middleware):
    async def before(self, state):
        # Pre-processing
        return state

    async def after(self, state):
        # Post-processing
        return state
```

### 6. RAG Integration
- **Vector Store Abstraction**: Chroma, Pinecone support
- **Embedding Models**: OpenAI, custom embeddings
- **Collection Management**: Multiple collections
- **Retrieval Scoring**: Similarity thresholds
- **RAG Manager**: Centralized RAG operations

**Example:**
```python
rag_config = RAGConfig(
    provider='chroma',
    embedding_model='openai/text-embedding-3-small'
)
manager = rag_config.setup()
```

### 7. MCP Integration
- **Server Management**: Automatic process lifecycle
- **Tool Discovery**: Dynamic tool registration
- **JSON-RPC Communication**: Standard protocol
- **Multiple Servers**: Slack, GitHub, custom servers
- **Environment Configuration**: Per-server env vars

**Example:**
```python
MCP_SERVERS = {
    'slack': {
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-slack']
    }
}
```

### 8. Authentication & Authorization
- **Auth Decorators**: `@requires_auth`, `@requires_permission`
- **Rate Limiting**: `@rate_limit` decorator
- **State Validation**: `@validate_state` decorator
- **AuthenticatedState**: Built-in auth state model
- **Permission System**: Role-based access control

**Example:**
```python
@requires_auth
@requires_permission('admin')
@rate_limit(max_requests=5, window=60)
async def admin_operation(self, state):
    pass
```

## 🛠️ CLI Features

### Setup & Installation
- **Automated Setup**: `langvel setup --with-venv`
- **Virtual Environment**: Automatic creation and activation
- **Dependency Installation**: One-command install
- **Progress UI**: Beautiful Rich-based progress indicators
- **Cross-Platform**: Windows, macOS, Linux support

### Generators
- **Agent Generator**: `langvel make:agent MyAgent`
- **State Generator**: `langvel make:state MyState`
- **Middleware Generator**: `langvel make:middleware MyMiddleware`
- **Tool Generator**: `langvel make:tool MyTool`
- **Template System**: Customizable templates

### Agent Management
- **List Agents**: `langvel agent list`
- **Test Agents**: `langvel agent test /path -i '{...}'`
- **Visualize Graphs**: `langvel agent graph /path -o graph.png`
- **Serve Agents**: `langvel agent serve --reload`
- **Hot Reload**: Automatic code reloading

### Development Tools
- **Rich Output**: Beautiful terminal output
- **Error Handling**: Helpful error messages
- **Auto-completion**: Shell completion support (future)
- **Debug Mode**: Verbose logging options

## 🌐 Server Features

### FastAPI Server
- **REST API**: Full HTTP API for agents
- **Streaming Support**: Server-sent events
- **CORS Handling**: Configurable CORS
- **Auto Documentation**: Swagger UI at /docs
- **Health Checks**: /health endpoint
- **Graph Visualization**: /agents/{path}/graph endpoint

### API Endpoints
```
GET  /                      # Root
GET  /health               # Health check
GET  /agents               # List all agents
POST /agents/{path}        # Invoke agent
GET  /agents/{path}/graph  # Get graph
```

### Request/Response Models
- **Pydantic Validation**: Type-safe requests
- **Streaming Response**: EventSource format
- **Error Handling**: Structured error responses
- **Metadata**: Rich response metadata

## 📊 State Management Features

### Checkpointers
- **Memory Checkpointer**: In-memory state (default)
- **PostgreSQL Checkpointer**: Persistent database storage
- **Redis Checkpointer**: Fast cache-based storage
- **Custom Checkpointers**: Easy to implement

### State Operations
- **State Updates**: Immutable update patterns
- **Message History**: Built-in conversation tracking
- **Context Management**: Arbitrary context data
- **Error Tracking**: Error state management

## 🔧 Configuration System

### Environment Variables
- **Type Safety**: Automatic type conversion
- **Defaults**: Sensible defaults
- **Validation**: Environment validation
- **Documentation**: Inline comments

### Config File
- **Python-based**: Full Python power
- **Hot Reload**: Development mode reloading
- **Modular**: Separate concerns
- **Extensible**: Easy to add custom config

## 🎨 Developer Experience

### Code Quality
- **Type Hints**: Full type coverage
- **Docstrings**: Comprehensive documentation
- **Examples**: Real-world examples
- **Tests**: Test coverage

### Documentation
- **README**: Comprehensive overview
- **INSTALL.md**: Detailed installation
- **QUICKSTART.md**: Rapid onboarding
- **FEATURES.md**: This file!
- **Inline Docs**: Well-commented code

### Templates
- **Agent Templates**: Ready-to-use patterns
- **State Templates**: Common state models
- **Middleware Templates**: Standard middleware
- **Tool Templates**: Tool patterns

## 🚀 Performance Features

### Efficiency
- **Lazy Loading**: Import only what's needed
- **Async/Await**: Full async support
- **Streaming**: Efficient data streaming
- **Caching**: Built-in caching support

### Scalability
- **Worker Support**: Multi-worker deployment
- **State Persistence**: Database-backed state
- **Load Balancing**: Ready for load balancers
- **Horizontal Scaling**: Stateless design

## 🔐 Security Features

### Authentication
- **API Key Support**: Environment-based keys
- **Session Management**: Built-in sessions
- **Permission System**: Role-based access
- **Rate Limiting**: Built-in rate limiting

### Best Practices
- **Input Validation**: Pydantic validation
- **Error Handling**: Safe error messages
- **Environment Isolation**: Virtual environments
- **Secret Management**: .env files (git-ignored)

## 🧪 Testing Features

### Test Support
- **Test Utilities**: Built-in test helpers
- **Mock Support**: Easy mocking
- **Setup Tests**: Verify installation
- **Integration Tests**: End-to-end testing

### CI/CD Ready
- **GitHub Actions**: Ready for CI
- **Docker Support**: (Coming soon)
- **Test Coverage**: Coverage reports
- **Linting**: Black, Ruff integration

## 📦 Distribution Features

### Package Management
- **pyproject.toml**: Modern Python packaging
- **Editable Install**: Development mode
- **Dependencies**: Well-defined dependencies
- **Version Management**: Semantic versioning

### Installation Options
- **PyPI**: (Coming soon)
- **Git Install**: Clone and install
- **Docker**: (Coming soon)
- **Pre-built Binaries**: (Future)

## 🎯 Use Case Support

### Built For
- **Customer Support**: RAG + routing + sentiment
- **Code Review**: LLM tools + GitHub integration
- **Data Analysis**: HTTP tools + streaming
- **Content Generation**: LLM + templating
- **Automation**: MCP + custom tools
- **Research**: RAG + knowledge bases

## 🔮 Coming Soon

### Roadmap Features
- [ ] Docker images
- [ ] PyPI distribution
- [ ] Web UI for agent management
- [ ] Plugin system
- [ ] Agent marketplace
- [ ] More MCP servers
- [ ] Enhanced monitoring
- [ ] Performance analytics
- [ ] Agent versioning
- [ ] A/B testing support

## 📊 Stats

- **40+ Files**: Core framework
- **3,500+ Lines**: Production-ready code
- **8 Core Modules**: Well-organized
- **20+ CLI Commands**: Full tooling
- **5 Tool Types**: Comprehensive tooling
- **3 Checkpointers**: Flexible persistence
- **10+ Built-in Middleware**: Ready to use

## 🎉 Summary

Langvel provides everything you need to build production-ready AI agents:
- ✅ Laravel-inspired DX
- ✅ Full LangGraph power
- ✅ Comprehensive tooling
- ✅ Production-ready
- ✅ Fully typed
- ✅ Well documented
- ✅ Easy to extend
- ✅ Active development

**Start building amazing agents today!** 🚀
