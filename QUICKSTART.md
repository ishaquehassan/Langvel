# Langvel Quick Start Guide

Get up and running with Langvel in under 5 minutes!

## ⚡ Quick Installation

### Option 1: Automated Setup (Fastest)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/langvel.git
cd langvel

# 2. Run setup script
python setup.py

# 3. Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 4. Add your API keys to .env
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx

# Done! 🎉
```

### Option 2: Using CLI (Most Beautiful)

```bash
# 1. Install Langvel
pip install -e .

# 2. Run setup with progress UI
langvel setup --with-venv

# 3. Activate venv and configure .env
source venv/bin/activate
nano .env
```

## 🎯 Create Your First Agent

### Step 1: Generate Files

```bash
# Create agent
langvel make:agent HelloAgent

# Create state model (optional but recommended)
langvel make:state HelloState
```

### Step 2: Define Your Agent

Edit `app/agents/helloagent.py`:

```python
from langvel.core.agent import Agent
from langvel.state.base import StateModel
from pydantic import Field

class HelloState(StateModel):
    name: str
    greeting: str = ""

class HelloAgent(Agent):
    state_model = HelloState

    def build_graph(self):
        return self.start().then(self.greet).end()

    async def greet(self, state: HelloState):
        state.greeting = f"Hello, {state.name}! Welcome to Langvel!"
        return state
```

### Step 3: Register Route

Edit `routes/agent.py`:

```python
from langvel.routing.router import router
from app.agents.helloagent import HelloAgent

@router.flow('/hello')
class HelloFlow(HelloAgent):
    pass
```

### Step 4: Test It!

**Option A: Via CLI**
```bash
langvel agent test /hello -i '{"name": "World"}'
```

**Option B: Via Server**
```bash
# Start server
langvel agent serve --reload

# In another terminal, test with curl
curl -X POST http://localhost:8000/agents/hello \
  -H "Content-Type: application/json" \
  -d '{"input": {"name": "World"}}'
```

**Option C: Via Python**
```python
import asyncio
from app.agents.helloagent import HelloAgent

async def main():
    agent = HelloAgent()
    result = await agent.invoke({"name": "World"})
    print(result.greeting)

asyncio.run(main())
```

## 🚀 Next Steps

### Add RAG (Retrieval Augmented Generation)

```python
from langvel.tools.decorators import rag_tool

class MyAgent(Agent):
    @rag_tool(collection='docs', k=5)
    async def search_docs(self, state):
        # Documents automatically retrieved and added to state
        return state
```

### Add Custom Tools

```python
from langvel.tools.decorators import tool

class MyAgent(Agent):
    @tool(description="Calculate sum")
    async def calculate(self, a: int, b: int) -> int:
        return a + b
```

### Add Middleware

```python
# Register in routes
@router.flow('/my-agent', middleware=['logging', 'auth', 'rate_limit'])
class MyAgentFlow(MyAgent):
    pass
```

### Add MCP Integration

```python
# Configure in config/langvel.py
MCP_SERVERS = {
    'slack': {
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-slack'],
        'env': {'SLACK_BOT_TOKEN': 'xoxb-xxxxx'}
    }
}

# Use in agent
from langvel.tools.decorators import mcp_tool

class MyAgent(Agent):
    @mcp_tool(server='slack', tool_name='send_message')
    async def notify(self, message: str):
        pass  # Automatically calls Slack
```

### Add Conditional Routing

```python
class MyAgent(Agent):
    def build_graph(self):
        return (
            self.start()
            .then(self.classify)
            .branch(
                {
                    'urgent': self.handle_urgent,
                    'normal': self.handle_normal,
                    'low': self.handle_low
                },
                condition_func=lambda state: state.priority
            )
            .merge(self.finalize)
            .end()
        )
```

### Add Authentication

```python
from langvel.auth.decorators import requires_auth, requires_permission

class MyAgent(Agent):
    @requires_auth
    async def sensitive_operation(self, state):
        # Only for authenticated users
        pass

    @requires_permission('admin')
    async def admin_operation(self, state):
        # Only for admins
        pass
```

## 📚 Common Commands

```bash
# Generate components
langvel make:agent MyAgent
langvel make:state MyState
langvel make:middleware MyMiddleware
langvel make:tool MyTool

# Manage agents
langvel agent list                          # List all agents
langvel agent test /path -i '{...}'         # Test agent
langvel agent graph /path -o graph.png      # Visualize
langvel agent serve                         # Start server
langvel agent serve --reload --port 8080    # Dev mode

# Help
langvel --help
langvel make --help
langvel agent --help
```

## 🔧 Configuration Tips

### Environment Variables (.env)

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx

# Optional
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_TEMPERATURE=0.7
STATE_CHECKPOINTER=memory
DEBUG=true
```

### Config File (config/langvel.py)

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Customize everything!
LLM_PROVIDER = 'anthropic'
LLM_MODEL = os.getenv('LLM_MODEL', 'claude-3-5-sonnet-20241022')
RAG_PROVIDER = 'chroma'
STATE_CHECKPOINTER = 'memory'  # or 'postgres', 'redis'
```

## 🐛 Troubleshooting

**Virtual environment not found:**
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

**Import errors:**
```bash
# Make sure you're in the venv
source venv/bin/activate
pip install -e .
```

**Port already in use:**
```bash
langvel agent serve --port 8080
```

**Module not found:**
```bash
# Reinstall in editable mode
pip install -e .
```

## 📖 Learn More

- **Full Documentation**: [README.md](./README.md)
- **Detailed Install Guide**: [INSTALL.md](./INSTALL.md)
- **Example Agents**: [app/agents/](./app/agents/)
- **Architecture**: See README.md#architecture

## 💡 Pro Tips

1. **Use --reload for development**: `langvel agent serve --reload`
2. **Test agents before deploying**: `langvel agent test /path`
3. **Visualize graphs**: `langvel agent graph /path -o graph.png`
4. **Use middleware for cross-cutting concerns**
5. **Check app/agents/ for real-world examples**
6. **Keep .env secure** (never commit it!)
7. **Use state models for type safety**

## 🎉 You're Ready!

You now know enough to build production-ready AI agents with Langvel!

**Happy building!** 🚀

Need help? Check out:
- [GitHub Issues](https://github.com/yourusername/langvel/issues)
- [Discord Community](https://discord.gg/langvel)
- [Documentation](https://langvel.dev)
