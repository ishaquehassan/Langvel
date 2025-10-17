# LLM Integration Guide

Complete guide to using LLMs in Langvel agents.

## 🎯 Overview

Langvel provides multiple ways to integrate LLMs into your agents:

1. **Direct Invocation** - `self.llm.invoke()` for simple queries
2. **Streaming** - `self.llm.stream()` for real-time responses
3. **Multi-turn Conversations** - `self.llm.chat()` for context-aware dialogs
4. **Structured Output** - Get Pydantic models back from LLMs
5. **LLM Tools** - `@llm_tool` decorator for tool-based LLM usage
6. **Convenience Functions** - `ask_llm()` and `stream_llm()` for quick usage

## 🚀 Quick Start

### Basic LLM Usage

Every agent automatically has an `self.llm` instance ready to use:

```python
from langvel.core.agent import Agent
from langvel.state.base import StateModel

class MyAgent(Agent):
    state_model = StateModel

    def build_graph(self):
        return self.start().then(self.process).end()

    async def process(self, state):
        # Simple LLM query
        response = await self.llm.invoke("What is Python?")
        state.add_message("assistant", response)
        return state
```

### With System Prompt

```python
async def process(self, state):
    response = await self.llm.invoke(
        prompt="Explain recursion",
        system_prompt="You are a patient teacher explaining to beginners."
    )
    return state
```

## 📖 Usage Methods

### 1. Direct Invocation

The most straightforward way to call an LLM:

```python
async def my_node(self, state):
    response = await self.llm.invoke(
        prompt=f"Analyze this: {state.query}",
        system_prompt="You are an expert analyst."
    )
    state.response = response
    return state
```

**When to use:**
- Simple, one-off LLM queries
- When you need direct control
- Quick prototyping

### 2. Streaming Responses

Stream responses in real-time:

```python
async def stream_response(self, state):
    result = ""
    async for chunk in self.llm.stream(
        prompt=f"Tell a story about {state.topic}",
        system_prompt="You are a creative storyteller."
    ):
        result += chunk
        # Could emit to client here
        print(chunk, end="", flush=True)

    state.story = result
    return state
```

**When to use:**
- Long-form content generation
- Real-time user feedback
- Progressive rendering

### 3. Multi-turn Conversations

Maintain context across multiple exchanges:

```python
async def chat(self, state):
    # Build conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."},
        {"role": "user", "content": "What can I build with it?"}
    ]

    response = await self.llm.chat(messages)
    state.add_message("assistant", response)
    return state
```

**When to use:**
- Conversational agents
- Context-aware responses
- Follow-up questions

### 4. Structured Output

Get Pydantic models back from LLMs:

```python
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1)
    keywords: List[str]
    summary: str

async def analyze(self, state):
    # Get structured LLM
    llm_structured = self.llm.with_structured_output(Analysis)

    # Invoke
    result: Analysis = await llm_structured.ainvoke(
        f"Analyze this text: {state.text}"
    )

    state.sentiment = result.sentiment
    state.confidence = result.confidence
    state.keywords = result.keywords

    return state
```

**When to use:**
- Need typed, validated output
- Extracting specific data
- Integrating with typed systems

### 5. Using @llm_tool Decorator

Mark methods as LLM-powered tools:

```python
from langvel.tools.decorators import llm_tool

class MyAgent(Agent):
    @llm_tool(system_prompt="You are a code reviewer")
    async def review_code(self, state):
        prompt = f"Review this code:\n{state.code}"
        review = await self.llm.invoke(prompt)
        state.review = review
        return state
```

**When to use:**
- Marking LLM operations as tools
- Metadata tracking
- Consistent LLM configuration

### 6. Convenience Functions

Quick LLM queries without agent setup:

```python
from langvel.llm.manager import ask_llm, stream_llm

# Simple query
result = await ask_llm("What is the capital of France?")

# With system prompt
result = await ask_llm(
    "Explain quantum computing",
    system_prompt="You are a physicist",
    temperature=0.3
)

# Streaming
async for chunk in stream_llm("Tell me a joke"):
    print(chunk, end="")
```

**When to use:**
- Quick scripts
- Testing
- Outside agent context

## ⚙️ Configuration

### Global Configuration

Set defaults in `config/langvel.py`:

```python
# LLM Configuration
LLM_PROVIDER = 'anthropic'  # or 'openai'
LLM_MODEL = 'claude-3-5-sonnet-20241022'
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096
```

### Per-Agent Configuration

Override in agent initialization:

```python
class MyAgent(Agent):
    def _init_llm(self):
        from langvel.llm.manager import LLMManager

        self.llm = LLMManager(
            provider='openai',
            model='gpt-4',
            temperature=0.5,
            max_tokens=2000
        )
```

### Per-Call Configuration

Override for specific calls:

```python
response = await self.llm.invoke(
    prompt="Be creative!",
    temperature=1.0,  # Override default
    max_tokens=500
)
```

## 🎨 Supported Providers

### Anthropic (Claude)

```python
from langvel.llm.manager import LLMManager

llm = LLMManager(
    provider='anthropic',
    model='claude-3-5-sonnet-20241022',
    temperature=0.7
)
```

**Models:**
- `claude-3-5-sonnet-20241022` (recommended)
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### OpenAI (GPT)

```python
llm = LLMManager(
    provider='openai',
    model='gpt-4',
    temperature=0.7
)
```

**Models:**
- `gpt-4` (recommended)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## 💡 Best Practices

### 1. Use System Prompts

Always provide context:

```python
response = await self.llm.invoke(
    prompt=user_query,
    system_prompt="You are an expert in {domain}. Be concise and accurate."
)
```

### 2. Handle Errors

Wrap LLM calls in try-catch:

```python
async def safe_llm_call(self, state):
    try:
        response = await self.llm.invoke(state.query)
        state.response = response
    except Exception as e:
        state.error = str(e)
        state.response = "I encountered an error. Please try again."
    return state
```

### 3. Use Structured Output for Data Extraction

Don't parse text when you can get structured data:

```python
# ❌ Bad: Parse text
response = await self.llm.invoke("Extract the date and amount")
# Now parse the text...

# ✅ Good: Structured output
class Transaction(BaseModel):
    date: str
    amount: float

llm_structured = self.llm.with_structured_output(Transaction)
result: Transaction = await llm_structured.ainvoke("...")
```

### 4. Stream Long Responses

For better UX:

```python
async def generate_article(self, state):
    article = ""
    async for chunk in self.llm.stream(f"Write about {state.topic}"):
        article += chunk
        # Emit chunk to client for progressive rendering
        await self.emit_chunk(chunk)
    return state
```

### 5. Reuse LLM Instance

Don't create new instances for each call:

```python
# ❌ Bad: Creates new instance each time
async def process(self, state):
    llm = LLMManager()  # Wasteful
    response = await llm.invoke("...")

# ✅ Good: Use self.llm
async def process(self, state):
    response = await self.llm.invoke("...")  # Reuses instance
```

## 📝 Complete Example

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from langvel.core.agent import Agent
from langvel.state.base import StateModel
from langvel.tools.decorators import llm_tool

# Define structured output
class EmailDraft(BaseModel):
    subject: str
    body: str
    tone: str = Field(description="formal, casual, or friendly")
    urgency: str = Field(description="low, medium, or high")

# Define state
class EmailAssistantState(StateModel):
    user_request: str
    draft: Optional[EmailDraft] = None
    refined_draft: Optional[str] = None

# Create agent
class EmailAssistantAgent(Agent):
    state_model = EmailAssistantState
    middleware = ['logging', 'rate_limit']

    def build_graph(self):
        return (
            self.start()
            .then(self.generate_draft)
            .then(self.refine_draft)
            .end()
        )

    async def generate_draft(self, state: EmailAssistantState):
        """Generate structured email draft."""
        llm_structured = self.llm.with_structured_output(EmailDraft)

        prompt = f"""
        Generate an email based on: {state.user_request}

        Consider appropriate tone and urgency level.
        """

        draft = await llm_structured.ainvoke(
            prompt,
            system_prompt="You are a professional email writing assistant."
        )

        state.draft = draft
        return state

    @llm_tool(system_prompt="You refine emails for clarity and impact.")
    async def refine_draft(self, state: EmailAssistantState):
        """Refine the draft for better clarity."""
        prompt = f"""
        Refine this email:
        Subject: {state.draft.subject}
        Body: {state.draft.body}

        Make it more {state.draft.tone} while maintaining the message.
        """

        refined = await self.llm.invoke(prompt)
        state.refined_draft = refined
        return state

# Usage
async def main():
    agent = EmailAssistantAgent()

    result = await agent.invoke({
        "user_request": "Write to my professor requesting an extension on my assignment"
    })

    print("Subject:", result.draft.subject)
    print("Tone:", result.draft.tone)
    print("Urgency:", result.draft.urgency)
    print("\nRefined Draft:")
    print(result.refined_draft)
```

## 🔍 Debugging

### Enable Verbose Logging

```python
import logging
logging.getLogger('langchain').setLevel(logging.DEBUG)
```

### Inspect LLM Calls

```python
# In development, log all LLM interactions
async def my_node(self, state):
    prompt = "Your prompt here"
    print(f"LLM Prompt: {prompt}")

    response = await self.llm.invoke(prompt)

    print(f"LLM Response: {response}")
    return state
```

## 📊 Performance Tips

1. **Batch similar queries** when possible
2. **Cache frequent responses** using Redis/Postgres
3. **Use cheaper models** for simple tasks (Claude Haiku, GPT-3.5)
4. **Limit max_tokens** to control costs
5. **Use streaming** for long responses
6. **Implement timeouts** for reliability

## 🔐 Security

1. **Never expose API keys** in code
2. **Sanitize user inputs** before sending to LLM
3. **Validate LLM outputs** before using
4. **Use rate limiting** to prevent abuse
5. **Log LLM usage** for monitoring

## 🚀 Next Steps

- Check out [app/agents/code_review_agent.py](../app/agents/code_review_agent.py) for complete examples
- Read about [RAG Integration](./RAG_GUIDE.md) for combining LLM with retrieval
- Explore [MCP Integration](./MCP_GUIDE.md) for external tools
- See [Agent Patterns](./AGENT_PATTERNS.md) for common use cases

---

**Happy building with LLMs!** 🤖
