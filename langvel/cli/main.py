"""Langvel CLI - Artisan-inspired command-line interface."""

import click
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from pathlib import Path

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    Langvel - A Laravel-inspired framework for LangGraph agents.

    Build powerful AI agents with elegant syntax.
    """
    pass


@cli.group()
def make():
    """Generate new components (agents, tools, middleware, etc.)."""
    pass


@make.command()
@click.argument('name')
@click.option('--state', default='StateModel', help='State model to use')
def agent(name: str, state: str):
    """Create a new agent."""
    console.print(f"[green]Creating agent:[/green] {name}")

    # Create agent file
    agent_content = f'''"""
{name} Agent
"""

from langvel.core.agent import Agent
from langvel.state.base import {state}
from langvel.tools.decorators import tool


class {name}(Agent):
    """
    {name} agent.

    Handles...
    """

    state_model = {state}
    middleware = []  # Add middleware like ['auth', 'rate_limit']

    def build_graph(self):
        """Define the agent workflow."""
        return (
            self.start()
            .then(self.process)
            .end()
        )

    async def process(self, state: {state}) -> {state}:
        """
        Main processing logic.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        # Your logic here
        state.add_message("assistant", "Response from {name}")
        return state

    @tool(description="Example custom tool")
    async def example_tool(self, input_text: str) -> str:
        """
        Example tool implementation.

        Args:
            input_text: Input text to process

        Returns:
            Processed text
        """
        return f"Processed: {{input_text}}"
'''

    # Write file
    agent_path = Path(f"app/agents/{name.lower()}.py")
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(agent_content)

    console.print(f"[green]✓[/green] Agent created at: {agent_path}")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("  1. Register the agent in routes/agent.py")
    console.print(f"  2. Implement the build_graph() method in {agent_path}")


@make.command()
@click.argument('name')
def tool(name: str):
    """Create a new tool."""
    console.print(f"[green]Creating tool:[/green] {name}")

    tool_content = f'''"""
{name} Tool
"""

from langvel.tools.decorators import tool


@tool(description="{name} tool")
async def {name.lower()}(input_data: str) -> str:
    """
    {name} implementation.

    Args:
        input_data: Input data

    Returns:
        Processed result
    """
    # Your implementation here
    return f"Result from {name}"
'''

    tool_path = Path(f"app/tools/{name.lower()}.py")
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text(tool_content)

    console.print(f"[green]✓[/green] Tool created at: {tool_path}")


@make.command()
@click.argument('name')
def middleware(name: str):
    """Create a new middleware."""
    console.print(f"[green]Creating middleware:[/green] {name}")

    middleware_content = f'''"""
{name} Middleware
"""

from typing import Any, Dict
from langvel.middleware.base import Middleware


class {name}(Middleware):
    """
    {name} middleware.

    Handles...
    """

    async def before(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute before agent runs.

        Args:
            state: Input state

        Returns:
            Modified state
        """
        # Your before logic here
        return state

    async def after(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute after agent runs.

        Args:
            state: Output state

        Returns:
            Modified state
        """
        # Your after logic here
        return state
'''

    middleware_path = Path(f"app/middleware/{name.lower()}.py")
    middleware_path.parent.mkdir(parents=True, exist_ok=True)
    middleware_path.write_text(middleware_content)

    console.print(f"[green]✓[/green] Middleware created at: {middleware_path}")


@make.command()
@click.argument('name')
def state(name: str):
    """Create a new state model."""
    console.print(f"[green]Creating state model:[/green] {name}")

    state_content = f'''"""
{name} State Model
"""

from typing import Optional
from pydantic import Field
from langvel.state.base import StateModel


class {name}(StateModel):
    """
    {name} state.

    Manages...
    """

    # Add your custom fields here
    query: str = Field(description="User query")
    response: Optional[str] = Field(default=None, description="Agent response")

    class Config:
        checkpointer = "memory"  # or "postgres", "redis"
        interrupts = []  # Add interrupt points like ['before_response']
'''

    state_path = Path(f"app/models/{name.lower()}.py")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state_content)

    console.print(f"[green]✓[/green] State model created at: {state_path}")


@cli.group()
def agent_cmd():
    """Agent management commands."""
    pass


@agent_cmd.command(name='list')
def list_agents():
    """List all registered agents."""
    try:
        import sys
        sys.path.insert(0, str(Path.cwd()))

        from routes.agent import router

        table = Table(title="Registered Agents")
        table.add_column("Path", style="cyan")
        table.add_column("Agent", style="green")
        table.add_column("Middleware", style="yellow")

        routes = router.list_routes()
        for route in routes:
            middleware_str = ", ".join(route['middleware']) if route['middleware'] else "-"
            table.add_row(
                route['path'],
                route['agent'],
                middleware_str
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


@agent_cmd.command(name='serve')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the Langvel agent server."""
    console.print(f"[green]Starting Langvel server on {host}:{port}[/green]")

    import uvicorn
    uvicorn.run(
        "langvel.server:app",
        host=host,
        port=port,
        reload=reload
    )


@agent_cmd.command(name='test')
@click.argument('agent_path')
@click.option('--input', '-i', 'input_data', help='Input JSON data')
def test_agent(agent_path: str, input_data: str):
    """Test an agent with sample input."""
    import asyncio
    import json
    import sys
    sys.path.insert(0, str(Path.cwd()))

    console.print(f"[green]Testing agent:[/green] {agent_path}")

    try:
        from routes.agent import router

        agent_class = router.get(agent_path)
        if not agent_class:
            console.print(f"[red]Error:[/red] Agent not found at path '{agent_path}'")
            return

        # Parse input
        if input_data:
            input_dict = json.loads(input_data)
        else:
            input_dict = {'query': 'test query'}

        # Run agent
        async def run():
            agent = agent_class()
            result = await agent.invoke(input_dict)
            return result

        result = asyncio.run(run())

        console.print("\n[green]Result:[/green]")
        console.print(Syntax(json.dumps(result, indent=2, default=str), "json"))

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())


@agent_cmd.command(name='graph')
@click.argument('agent_path')
@click.option('--output', '-o', default='graph.png', help='Output file path')
def visualize_graph(agent_path: str, output: str):
    """Visualize an agent's graph."""
    import sys
    sys.path.insert(0, str(Path.cwd()))

    console.print(f"[green]Generating graph for:[/green] {agent_path}")

    try:
        from routes.agent import router

        agent_class = router.get(agent_path)
        if not agent_class:
            console.print(f"[red]Error:[/red] Agent not found at path '{agent_path}'")
            return

        agent = agent_class()
        graph = agent.compile()

        # Generate PNG
        png_data = graph.get_graph().draw_mermaid_png()

        # Save to file
        output_path = Path(output)
        output_path.write_bytes(png_data)

        console.print(f"[green]✓[/green] Graph saved to: {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


@cli.command()
@click.option('--with-venv', is_flag=True, help='Also setup virtual environment and install dependencies')
def setup(with_venv: bool):
    """
    Complete setup of Langvel framework.

    Creates virtual environment, installs dependencies, and initializes project.
    """
    import subprocess
    import sys
    import os
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console.print("[bold green]🚀 Langvel Framework Setup[/bold green]\n")

    if with_venv:
        # Step 1: Create virtual environment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task1 = progress.add_task("[cyan]Creating virtual environment...", total=None)

            venv_path = Path.cwd() / "venv"

            if venv_path.exists():
                console.print("[yellow]⚠[/yellow]  Virtual environment already exists")
            else:
                try:
                    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
                    console.print("[green]✓[/green] Virtual environment created")
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]✗[/red] Failed to create virtual environment: {e}")
                    return

            progress.remove_task(task1)

        # Step 2: Determine activation script
        if sys.platform == "win32":
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_executable = venv_path / "Scripts" / "pip"
            python_executable = venv_path / "Scripts" / "python"
        else:
            activate_script = venv_path / "bin" / "activate"
            pip_executable = venv_path / "bin" / "pip"
            python_executable = venv_path / "bin" / "python"

        # Step 3: Upgrade pip
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task2 = progress.add_task("[cyan]Upgrading pip...", total=None)

            try:
                subprocess.run(
                    [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
                    check=True,
                    capture_output=True
                )
                console.print("[green]✓[/green] Pip upgraded")
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]⚠[/yellow]  Pip upgrade warning (continuing anyway)")

            progress.remove_task(task2)

        # Step 4: Install dependencies
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task3 = progress.add_task("[cyan]Installing Langvel and dependencies...", total=None)

            try:
                # Install in editable mode
                subprocess.run(
                    [str(pip_executable), "install", "-e", "."],
                    check=True,
                    capture_output=True
                )
                console.print("[green]✓[/green] Dependencies installed")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]✗[/red] Failed to install dependencies: {e}")
                return

            progress.remove_task(task3)

    # Step 5: Initialize project structure
    console.print("\n[cyan]Initializing project structure...[/cyan]")

    # Create directory structure
    dirs = [
        'app/agents',
        'app/middleware',
        'app/tools',
        'app/models',
        'config',
        'routes',
        'storage/logs',
        'storage/checkpoints',
        'tests'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create config file
    config_content = '''"""
Langvel Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')
LLM_MODEL = os.getenv('LLM_MODEL', 'claude-3-5-sonnet-20241022')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))

# RAG Configuration
RAG_PROVIDER = os.getenv('RAG_PROVIDER', 'chroma')
RAG_EMBEDDING_MODEL = os.getenv('RAG_EMBEDDING_MODEL', 'openai/text-embedding-3-small')

# MCP Servers
MCP_SERVERS = {
    # Example: 'slack': {
    #     'command': 'npx',
    #     'args': ['-y', '@modelcontextprotocol/server-slack'],
    #     'env': {'SLACK_BOT_TOKEN': os.getenv('SLACK_BOT_TOKEN')}
    # }
}

# State Configuration
STATE_CHECKPOINTER = os.getenv('STATE_CHECKPOINTER', 'memory')  # memory, postgres, redis

# Database (for postgres checkpointer)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/langvel')

# Redis (for redis checkpointer)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
'''

    if not Path('config/langvel.py').exists():
        Path('config/langvel.py').write_text(config_content)
        console.print("[green]✓[/green] Config file created")
    else:
        console.print("[yellow]⚠[/yellow]  Config file already exists")

    # Create routes file
    routes_content = '''"""
Agent Routes

Register your agents here.
"""

from langvel.routing.router import AgentRouter

router = AgentRouter()

# Example:
# @router.flow('/example', middleware=['logging'])
# class ExampleAgent(Agent):
#     def build_graph(self):
#         return self.start().then(self.handle).end()
'''

    if not Path('routes/agent.py').exists():
        Path('routes/agent.py').write_text(routes_content)
        console.print("[green]✓[/green] Routes file created")
    else:
        console.print("[yellow]⚠[/yellow]  Routes file already exists")

    # Create .env file
    env_content = '''# Langvel Environment Variables

# LLM Configuration
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_api_key_here

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_key_here

# State Management
STATE_CHECKPOINTER=memory

# Database (if using postgres)
# DATABASE_URL=postgresql://localhost/langvel

# Redis (if using redis)
# REDIS_URL=redis://localhost:6379
'''

    if not Path('.env').exists():
        Path('.env').write_text(env_content)
        console.print("[green]✓[/green] Environment file created")
    else:
        console.print("[yellow]⚠[/yellow]  Environment file already exists")

    # Create __init__.py files
    for dir_path in ['app', 'app/agents', 'app/middleware', 'app/tools', 'app/models']:
        init_file = Path(dir_path) / '__init__.py'
        if not init_file.exists():
            init_file.touch()

    console.print("\n[bold green]✨ Setup complete![/bold green]\n")

    if with_venv:
        console.print("[yellow]To activate the virtual environment:[/yellow]")
        if sys.platform == "win32":
            console.print("  [cyan]venv\\Scripts\\activate[/cyan]")
        else:
            console.print("  [cyan]source venv/bin/activate[/cyan]")
        console.print()

    console.print("[yellow]Next steps:[/yellow]")
    console.print("  1. Update .env with your API keys")
    console.print("  2. Create your first agent: [cyan]langvel make:agent MyAgent[/cyan]")
    console.print("  3. Register it in routes/agent.py")
    console.print("  4. Start the server: [cyan]langvel agent serve[/cyan]")


@cli.command()
def init():
    """Initialize a new Langvel project (without venv setup)."""
    console.print("[green]Initializing Langvel project...[/green]")

    # Create directory structure
    dirs = [
        'app/agents',
        'app/middleware',
        'app/tools',
        'app/models',
        'config',
        'routes',
        'storage/logs',
        'storage/checkpoints',
        'tests'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create config file
    config_content = '''"""
Langvel Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')
LLM_MODEL = os.getenv('LLM_MODEL', 'claude-3-5-sonnet-20241022')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))

# RAG Configuration
RAG_PROVIDER = os.getenv('RAG_PROVIDER', 'chroma')
RAG_EMBEDDING_MODEL = os.getenv('RAG_EMBEDDING_MODEL', 'openai/text-embedding-3-small')

# MCP Servers
MCP_SERVERS = {
    # Example: 'slack': {
    #     'command': 'npx',
    #     'args': ['-y', '@modelcontextprotocol/server-slack'],
    #     'env': {'SLACK_BOT_TOKEN': os.getenv('SLACK_BOT_TOKEN')}
    # }
}

# State Configuration
STATE_CHECKPOINTER = os.getenv('STATE_CHECKPOINTER', 'memory')  # memory, postgres, redis

# Database (for postgres checkpointer)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/langvel')

# Redis (for redis checkpointer)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
'''

    Path('config/langvel.py').write_text(config_content)

    # Create routes file
    routes_content = '''"""
Agent Routes

Register your agents here.
"""

from langvel.routing.router import AgentRouter

router = AgentRouter()

# Example:
# @router.flow('/example', middleware=['logging'])
# class ExampleAgent(Agent):
#     def build_graph(self):
#         return self.start().then(self.handle).end()
'''

    Path('routes/agent.py').write_text(routes_content)

    # Create .env file
    env_content = '''# Langvel Environment Variables

# LLM Configuration
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_api_key_here

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_key_here

# State Management
STATE_CHECKPOINTER=memory

# Database (if using postgres)
# DATABASE_URL=postgresql://localhost/langvel

# Redis (if using redis)
# REDIS_URL=redis://localhost:6379
'''

    if not Path('.env').exists():
        Path('.env').write_text(env_content)

    console.print("[green]✓[/green] Project initialized!")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("  1. Update .env with your API keys")
    console.print("  2. Create your first agent: langvel make:agent MyAgent")
    console.print("  3. Register it in routes/agent.py")
    console.print("  4. Start the server: langvel agent serve")


if __name__ == '__main__':
    cli()
