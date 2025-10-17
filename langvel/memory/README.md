# Langvel Memory Systems

Complete memory implementation for Langvel agents providing semantic, episodic, and working memory capabilities.

## Overview

Langvel's memory system allows agents to:
- **Remember facts** about users across sessions (semantic memory)
- **Maintain conversation history** (episodic memory)
- **Track current context** (working memory)
- **Automatically extract entities** and relationships

## Memory Types

### 1. Semantic Memory (Long-Term)
Stores facts, entities, and relationships that persist across sessions.

**Backend:** PostgreSQL (for structured queries) + Vector Store (for semantic search)

**Use Cases:**
- User preferences and profile information
- Domain knowledge
- Learned facts from conversations

**Example:**
```python
# Store facts
await memory.semantic.store_fact(
    user_id='user123',
    fact='Works at TechCorp as a software engineer',
    metadata={'confidence': 0.95, 'source': 'conversation'}
)

# Store entities
await memory.semantic.store_entity(
    user_id='user123',
    entity_type='company',
    entity_name='TechCorp',
    properties={'industry': 'technology'}
)

# Store relationships
await memory.semantic.store_relationship(
    user_id='user123',
    subject='John Doe',
    relationship='works_at',
    object_entity='TechCorp'
)

# Recall with semantic search
facts = await memory.semantic.recall_facts(
    user_id='user123',
    query='Where does the user work?',
    limit=5
)
```

### 2. Episodic Memory (Short-Term)
Stores conversation turns with automatic pruning.

**Backend:** Redis (fast access with TTL)

**Use Cases:**
- Recent conversation context
- Session-specific information
- Temporary data

**Example:**
```python
# Add conversation turns
await memory.episodic.add_turn(
    session_id='session123',
    role='user',
    content='Hello, how are you?'
)

# Get recent conversation
recent = await memory.episodic.get_recent('session123', limit=10)

# Get conversation summary
summary = await memory.episodic.get_summary('session123', llm)

# Get context window (token-aware)
context = await memory.episodic.get_context_window(
    session_id='session123',
    max_tokens=2000
)
```

### 3. Working Memory (Current Context)
In-memory storage for current task with automatic pruning.

**Backend:** In-memory (OrderedDict with LRU)

**Use Cases:**
- Current task variables
- Temporary calculations
- Session state

**Example:**
```python
# Add items with priority
memory.working.add('user_intent', 'book_flight', priority=10)
memory.working.add('destination', 'New York', priority=5)

# Get items
intent = memory.working.get('user_intent')

# Convert to context string
context = memory.working.to_context_string()
# Output: "user_intent: book_flight\ndestination: New York"

# Auto-pruning when exceeding max_tokens
memory.working.set_max_tokens(4000)  # Keeps size manageable
```

## Unified Memory Manager

The `MemoryManager` provides a single interface for all memory types.

### Basic Usage

```python
from langvel.memory import MemoryManager

# Create manager
memory = MemoryManager()
await memory.initialize()

# Store information (automatically routes to appropriate memory)
await memory.remember(
    user_id='user123',
    session_id='session456',
    content='My name is Alice and I work at TechCorp',
    role='user',
    memory_type='auto'  # Detects facts automatically
)

# Recall information from all memory types
memories = await memory.recall(
    user_id='user123',
    session_id='session456',
    query='Tell me about myself'
)

# Build context for LLM
context = await memory.build_context(
    user_id='user123',
    session_id='session456',
    query='What do you know about me?',
    max_tokens=2000
)
```

### Automatic Fact Detection

The memory manager automatically detects and extracts:
- Names: "My name is Alice"
- Companies: "I work at TechCorp"
- Locations: "I live in San Francisco"
- Roles: "I work as a software engineer"

### Using Memory in Agents

```python
from langvel.core.agent import Agent
from langvel.state.base import StateModel

class MyAgent(Agent):
    enable_memory = True  # Enable memory systems

    def build_graph(self):
        return (
            self.start()
            .then(self.recall_memories)
            .then(self.process_with_context)
            .then(self.store_interaction)
            .end()
        )

    async def recall_memories(self, state):
        # Get memory context
        context = await self.memory_manager.build_context(
            user_id=state.user_id,
            session_id=state.session_id,
            query=state.query
        )
        state.context = context
        return state

    async def process_with_context(self, state):
        # Use memory in LLM prompt
        prompt = f"{state.context}\n\nUser: {state.query}"
        response = await self.llm.invoke(prompt)
        state.response = response
        return state

    async def store_interaction(self, state):
        # Store conversation in memory
        await self.memory_manager.remember(
            user_id=state.user_id,
            session_id=state.session_id,
            content=state.query,
            role='user'
        )

        await self.memory_manager.remember(
            user_id=state.user_id,
            session_id=state.session_id,
            content=state.response,
            role='assistant'
        )
        return state
```

## Configuration

### Environment Variables

```bash
# Memory backends
MEMORY_SEMANTIC_BACKEND=postgres  # or 'memory' for testing
MEMORY_EPISODIC_BACKEND=redis     # or 'memory' for testing

# Episodic memory settings
MEMORY_EPISODIC_TTL=86400  # 24 hours in seconds

# Working memory settings
MEMORY_WORKING_MAX_TOKENS=4000

# Automatic fact detection
MEMORY_AUTO_DETECT_FACTS=true
```

### Python Configuration

```python
# config/langvel.py
MEMORY_SEMANTIC_BACKEND = 'postgres'
MEMORY_EPISODIC_BACKEND = 'redis'
MEMORY_EPISODIC_TTL = 86400
MEMORY_WORKING_MAX_TOKENS = 4000
MEMORY_AUTO_DETECT_FACTS = True
```

## CLI Commands

Langvel provides CLI commands for memory management:

### Clear Memory
```bash
# Clear all memory for a user
langvel memory clear user123 --all

# Clear only semantic memory
langvel memory clear user123 --semantic

# Clear episodic memory for a session
langvel memory clear user123 --session session456 --episodic

# Clear working memory
langvel memory clear user123 --working
```

### Memory Statistics
```bash
# Get memory stats for a user
langvel memory stats user123

# Get stats for a specific session
langvel memory stats user123 --session session456
```

### List Memory
```bash
# List facts for a user
langvel memory list user123 --type facts --limit 20

# List active sessions
langvel memory list user123 --type sessions
```

## Database Setup

### PostgreSQL (Semantic Memory)

The semantic memory module automatically creates required tables:

```sql
-- Facts table
CREATE TABLE semantic_facts (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Entities table
CREATE TABLE semantic_entities (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, entity_type, entity_name)
);

-- Relationships table
CREATE TABLE semantic_relationships (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject_entity TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    object_entity TEXT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Redis (Episodic Memory)

No setup required. Keys are automatically created:

```
episodes:{session_id}  # List of conversation turns
```

## Advanced Features

### Custom Fact Extraction

```python
# Extend MemoryManager with custom extraction
class CustomMemoryManager(MemoryManager):
    async def _extract_and_store_facts(self, user_id, content, metadata):
        # Your custom extraction logic
        # Use NLP, NER, or LLM for extraction
        ...
```

### Memory Snapshots

```python
# Working memory snapshots
snapshot = memory.working.snapshot()

# Restore from snapshot
memory.working.restore(snapshot)
```

### Memory Merging

```python
# Merge two working memories
memory1.working.merge(memory2.working, strategy='higher_priority')
```

### Context Management

```python
# Get conversation context that fits in token budget
context = await memory.episodic.get_context_window(
    session_id='session123',
    max_tokens=2000
)

# Build structured context for LLM
context = await memory.build_context(
    user_id='user123',
    session_id='session456',
    format_style='detailed'  # or 'simple'
)
```

## Performance Considerations

### Semantic Memory (PostgreSQL)
- **Indexes**: User ID, entity type indexed for fast queries
- **Connection pooling**: Max 10 connections by default
- **Vector store**: Optional for semantic search

### Episodic Memory (Redis)
- **TTL**: Automatic cleanup after 24 hours (configurable)
- **Max turns**: Limited to 100 turns per session (configurable)
- **Async operations**: Non-blocking throughout

### Working Memory (In-Memory)
- **Auto-pruning**: LRU eviction when exceeding max tokens
- **Priority-based**: Higher priority items kept longer
- **Token estimation**: Rough approximation (1 token ≈ 4 chars)

## Testing

### With Real Databases

```python
import asyncio
from langvel.memory import MemoryManager

async def test_memory():
    memory = MemoryManager()
    await memory.initialize()

    # Your tests...

    await memory.close()

asyncio.run(test_memory())
```

### With In-Memory Backend (Testing)

```python
from langvel.memory import SemanticMemory, EpisodicMemory, MemoryManager

# Use memory backends (no database required)
semantic = SemanticMemory(backend='memory')
episodic = EpisodicMemory(backend='memory')

memory = MemoryManager(semantic=semantic, episodic=episodic)
await memory.initialize()

# Now test without PostgreSQL/Redis
```

## Examples

See `/app/agents/memory_agent.py` for a complete example demonstrating:
- Semantic memory for user facts
- Episodic memory for conversation
- Working memory for current context
- Automatic fact extraction
- Context-aware LLM responses

## Migration from v0.2.0

Memory systems are **new in v0.3.0**. No migration needed.

## Future Enhancements

Planned for future versions:
- **Semantic search**: Integration with vector stores (Chroma, Pinecone)
- **Memory compression**: Automatic summarization of old conversations
- **Memory analytics**: Usage patterns and insights
- **Shared memory**: Cross-user knowledge bases
- **Memory triggers**: Event-driven memory updates

## Troubleshooting

### "PostgreSQL connection failed"
- Ensure PostgreSQL is running
- Check `DATABASE_URL` in config
- Verify network connectivity

### "Redis connection failed"
- Ensure Redis is running
- Check `REDIS_URL` in config
- Verify Redis is accessible

### "Memory not persisting"
- Check that `enable_memory = True` in agent
- Verify `await memory.initialize()` is called
- Ensure database connections are valid

### "Out of memory"
- Reduce `MEMORY_WORKING_MAX_TOKENS`
- Implement memory compression
- Use memory cleanup commands

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/langvel/issues
- Documentation: https://langvel.dev/memory
- Examples: `/app/agents/memory_agent.py`
