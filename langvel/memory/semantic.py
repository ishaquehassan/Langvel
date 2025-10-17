"""Semantic memory - Long-term entity and fact storage."""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from langvel.logging import get_logger

logger = get_logger(__name__)


class SemanticMemory:
    """
    Long-term semantic memory for entities, facts, and relationships.

    Stores structured information that persists across sessions.
    Combines vector search (semantic) with SQL (structured queries).

    Example:
        memory = SemanticMemory(backend='postgres')

        # Store facts
        await memory.store_fact(
            user_id='user123',
            fact='Works at Acme Corp as a data scientist',
            metadata={'source': 'conversation', 'confidence': 0.95}
        )

        # Store entities
        await memory.store_entity(
            user_id='user123',
            entity_type='company',
            entity_name='Acme Corp',
            properties={'industry': 'technology', 'size': 'large'}
        )

        # Recall facts
        facts = await memory.recall_facts(
            user_id='user123',
            query='Where does the user work?',
            limit=5
        )
    """

    def __init__(
        self,
        backend: str = 'postgres',
        vector_store: Optional[Any] = None,
        connection_string: Optional[str] = None
    ):
        """
        Initialize semantic memory.

        Args:
            backend: Storage backend ('postgres', 'sqlite', 'memory')
            vector_store: Optional vector store for semantic search
            connection_string: Optional database connection string
        """
        self.backend = backend
        self.vector_store = vector_store
        self.connection_string = connection_string
        self._db = None
        self._initialized = False

        logger.info(
            "Semantic memory initialized",
            extra={'backend': backend, 'has_vector_store': vector_store is not None}
        )

    async def initialize(self):
        """Initialize database connection and create tables."""
        if self._initialized:
            return

        if self.backend == 'postgres':
            await self._init_postgres()
        elif self.backend == 'memory':
            await self._init_memory()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self._initialized = True
        logger.info("Semantic memory storage initialized", extra={'backend': self.backend})

    async def _init_postgres(self):
        """Initialize PostgreSQL backend."""
        import asyncpg
        from config.langvel import config

        conn_string = self.connection_string or config.DATABASE_URL

        try:
            self._db = await asyncpg.create_pool(
                conn_string,
                min_size=2,
                max_size=10
            )

            # Create tables
            async with self._db.acquire() as conn:
                # Facts table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS semantic_facts (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        fact TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                ''')

                # Entities table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS semantic_entities (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_name TEXT NOT NULL,
                        properties JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(user_id, entity_type, entity_name)
                    )
                ''')

                # Relationships table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS semantic_relationships (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        subject_entity TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        object_entity TEXT NOT NULL,
                        properties JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                ''')

                # Create indexes
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_semantic_facts_user
                    ON semantic_facts(user_id)
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_semantic_entities_user
                    ON semantic_entities(user_id)
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_semantic_entities_type
                    ON semantic_entities(entity_type)
                ''')

            logger.info("PostgreSQL semantic memory tables created")

        except Exception as e:
            logger.error(
                "Failed to initialize PostgreSQL semantic memory",
                extra={'error': str(e)},
                exc_info=True
            )
            raise

    async def _init_memory(self):
        """Initialize in-memory backend (for testing)."""
        self._db = {
            'facts': [],
            'entities': [],
            'relationships': []
        }
        logger.info("In-memory semantic storage initialized")

    async def store_fact(
        self,
        user_id: str,
        fact: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a fact about the user.

        Args:
            user_id: User identifier
            fact: Fact to store (e.g., "Works at Acme Corp")
            metadata: Optional metadata (source, confidence, etc.)

        Returns:
            Fact ID
        """
        if not self._initialized:
            await self.initialize()

        metadata = metadata or {}

        try:
            if self.backend == 'postgres':
                async with self._db.acquire() as conn:
                    row = await conn.fetchrow('''
                        INSERT INTO semantic_facts (user_id, fact, metadata)
                        VALUES ($1, $2, $3)
                        RETURNING id
                    ''', user_id, fact, json.dumps(metadata))
                    fact_id = row['id']

            elif self.backend == 'memory':
                fact_id = len(self._db['facts']) + 1
                self._db['facts'].append({
                    'id': fact_id,
                    'user_id': user_id,
                    'fact': fact,
                    'metadata': metadata,
                    'created_at': datetime.utcnow().isoformat()
                })

            # Store in vector store for semantic search
            if self.vector_store:
                await self.vector_store.aadd_documents([{
                    'content': fact,
                    'metadata': {
                        'user_id': user_id,
                        'type': 'fact',
                        'fact_id': fact_id,
                        **metadata
                    }
                }])

            logger.info(
                "Fact stored",
                extra={'user_id': user_id, 'fact_id': fact_id, 'fact_length': len(fact)}
            )

            return fact_id

        except Exception as e:
            logger.error(
                "Failed to store fact",
                extra={'user_id': user_id, 'error': str(e)},
                exc_info=True
            )
            raise

    async def store_entity(
        self,
        user_id: str,
        entity_type: str,
        entity_name: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store an entity (person, company, product, etc.).

        Args:
            user_id: User identifier
            entity_type: Type of entity (person, company, product, etc.)
            entity_name: Name of the entity
            properties: Entity properties

        Returns:
            Entity ID
        """
        if not self._initialized:
            await self.initialize()

        properties = properties or {}

        try:
            if self.backend == 'postgres':
                async with self._db.acquire() as conn:
                    # Upsert entity
                    row = await conn.fetchrow('''
                        INSERT INTO semantic_entities (user_id, entity_type, entity_name, properties)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (user_id, entity_type, entity_name)
                        DO UPDATE SET properties = $4, updated_at = NOW()
                        RETURNING id
                    ''', user_id, entity_type, entity_name, json.dumps(properties))
                    entity_id = row['id']

            elif self.backend == 'memory':
                # Find existing entity
                existing = next(
                    (e for e in self._db['entities']
                     if e['user_id'] == user_id
                     and e['entity_type'] == entity_type
                     and e['entity_name'] == entity_name),
                    None
                )

                if existing:
                    existing['properties'] = properties
                    existing['updated_at'] = datetime.utcnow().isoformat()
                    entity_id = existing['id']
                else:
                    entity_id = len(self._db['entities']) + 1
                    self._db['entities'].append({
                        'id': entity_id,
                        'user_id': user_id,
                        'entity_type': entity_type,
                        'entity_name': entity_name,
                        'properties': properties,
                        'created_at': datetime.utcnow().isoformat()
                    })

            logger.info(
                "Entity stored",
                extra={
                    'user_id': user_id,
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'entity_name': entity_name
                }
            )

            return entity_id

        except Exception as e:
            logger.error(
                "Failed to store entity",
                extra={'user_id': user_id, 'entity_type': entity_type, 'error': str(e)},
                exc_info=True
            )
            raise

    async def store_relationship(
        self,
        user_id: str,
        subject: str,
        relationship: str,
        object_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a relationship between entities.

        Args:
            user_id: User identifier
            subject: Subject entity name
            relationship: Relationship type (works_at, lives_in, etc.)
            object_entity: Object entity name
            properties: Relationship properties

        Returns:
            Relationship ID

        Example:
            await memory.store_relationship(
                user_id='user123',
                subject='John Doe',
                relationship='works_at',
                object_entity='Acme Corp',
                properties={'since': '2020', 'role': 'engineer'}
            )
        """
        if not self._initialized:
            await self.initialize()

        properties = properties or {}

        try:
            if self.backend == 'postgres':
                async with self._db.acquire() as conn:
                    row = await conn.fetchrow('''
                        INSERT INTO semantic_relationships
                        (user_id, subject_entity, relationship_type, object_entity, properties)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id
                    ''', user_id, subject, relationship, object_entity, json.dumps(properties))
                    rel_id = row['id']

            elif self.backend == 'memory':
                rel_id = len(self._db['relationships']) + 1
                self._db['relationships'].append({
                    'id': rel_id,
                    'user_id': user_id,
                    'subject_entity': subject,
                    'relationship_type': relationship,
                    'object_entity': object_entity,
                    'properties': properties,
                    'created_at': datetime.utcnow().isoformat()
                })

            logger.info(
                "Relationship stored",
                extra={
                    'user_id': user_id,
                    'relationship_id': rel_id,
                    'relationship': f"{subject} {relationship} {object_entity}"
                }
            )

            return rel_id

        except Exception as e:
            logger.error(
                "Failed to store relationship",
                extra={'user_id': user_id, 'error': str(e)},
                exc_info=True
            )
            raise

    async def recall_facts(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recall facts, optionally with semantic search.

        Args:
            user_id: User identifier
            query: Optional semantic search query
            limit: Maximum number of facts to return

        Returns:
            List of facts with metadata
        """
        if not self._initialized:
            await self.initialize()

        try:
            if query and self.vector_store:
                # Semantic search
                results = await self.vector_store.asimilarity_search(
                    query,
                    filter={'user_id': user_id, 'type': 'fact'},
                    k=limit
                )
                facts = [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': getattr(doc, 'score', None)
                    }
                    for doc in results
                ]

            elif self.backend == 'postgres':
                async with self._db.acquire() as conn:
                    rows = await conn.fetch('''
                        SELECT id, fact, metadata, created_at
                        FROM semantic_facts
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    ''', user_id, limit)

                    facts = [
                        {
                            'id': row['id'],
                            'content': row['fact'],
                            'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'],
                            'created_at': row['created_at'].isoformat()
                        }
                        for row in rows
                    ]

            elif self.backend == 'memory':
                user_facts = [
                    f for f in self._db['facts']
                    if f['user_id'] == user_id
                ]
                facts = sorted(user_facts, key=lambda x: x['created_at'], reverse=True)[:limit]

            logger.info(
                "Facts recalled",
                extra={'user_id': user_id, 'count': len(facts), 'with_query': query is not None}
            )

            return facts

        except Exception as e:
            logger.error(
                "Failed to recall facts",
                extra={'user_id': user_id, 'error': str(e)},
                exc_info=True
            )
            return []

    async def get_entity(
        self,
        user_id: str,
        entity_name: str,
        entity_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get entity by name.

        Args:
            user_id: User identifier
            entity_name: Entity name
            entity_type: Optional entity type to filter by

        Returns:
            Entity data or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.backend == 'postgres':
                async with self._db.acquire() as conn:
                    if entity_type:
                        row = await conn.fetchrow('''
                            SELECT * FROM semantic_entities
                            WHERE user_id = $1 AND entity_name = $2 AND entity_type = $3
                        ''', user_id, entity_name, entity_type)
                    else:
                        row = await conn.fetchrow('''
                            SELECT * FROM semantic_entities
                            WHERE user_id = $1 AND entity_name = $2
                        ''', user_id, entity_name)

                    if row:
                        return {
                            'id': row['id'],
                            'user_id': row['user_id'],
                            'entity_type': row['entity_type'],
                            'entity_name': row['entity_name'],
                            'properties': json.loads(row['properties']) if isinstance(row['properties'], str) else row['properties'],
                            'created_at': row['created_at'].isoformat()
                        }

            elif self.backend == 'memory':
                for entity in self._db['entities']:
                    if (entity['user_id'] == user_id and
                        entity['entity_name'] == entity_name and
                        (not entity_type or entity['entity_type'] == entity_type)):
                        return entity

            return None

        except Exception as e:
            logger.error(
                "Failed to get entity",
                extra={'user_id': user_id, 'entity_name': entity_name, 'error': str(e)},
                exc_info=True
            )
            return None

    async def get_relationships(
        self,
        user_id: str,
        subject: Optional[str] = None,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a subject or by type.

        Args:
            user_id: User identifier
            subject: Optional subject entity to filter by
            relationship_type: Optional relationship type to filter by

        Returns:
            List of relationships
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.backend == 'postgres':
                async with self._db.acquire() as conn:
                    query = 'SELECT * FROM semantic_relationships WHERE user_id = $1'
                    params = [user_id]

                    if subject:
                        query += ' AND subject_entity = $2'
                        params.append(subject)
                    if relationship_type:
                        query += f' AND relationship_type = ${len(params) + 1}'
                        params.append(relationship_type)

                    rows = await conn.fetch(query, *params)

                    return [
                        {
                            'id': row['id'],
                            'subject': row['subject_entity'],
                            'relationship': row['relationship_type'],
                            'object': row['object_entity'],
                            'properties': json.loads(row['properties']) if isinstance(row['properties'], str) else row['properties'],
                            'created_at': row['created_at'].isoformat()
                        }
                        for row in rows
                    ]

            elif self.backend == 'memory':
                relationships = [
                    r for r in self._db['relationships']
                    if r['user_id'] == user_id and
                    (not subject or r['subject_entity'] == subject) and
                    (not relationship_type or r['relationship_type'] == relationship_type)
                ]
                return relationships

            return []

        except Exception as e:
            logger.error(
                "Failed to get relationships",
                extra={'user_id': user_id, 'error': str(e)},
                exc_info=True
            )
            return []

    async def clear_user_memory(self, user_id: str):
        """
        Clear all memory for a user.

        Args:
            user_id: User identifier
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.backend == 'postgres':
                async with self._db.acquire() as conn:
                    await conn.execute('DELETE FROM semantic_facts WHERE user_id = $1', user_id)
                    await conn.execute('DELETE FROM semantic_entities WHERE user_id = $1', user_id)
                    await conn.execute('DELETE FROM semantic_relationships WHERE user_id = $1', user_id)

            elif self.backend == 'memory':
                self._db['facts'] = [f for f in self._db['facts'] if f['user_id'] != user_id]
                self._db['entities'] = [e for e in self._db['entities'] if e['user_id'] != user_id]
                self._db['relationships'] = [r for r in self._db['relationships'] if r['user_id'] != user_id]

            logger.info("User memory cleared", extra={'user_id': user_id})

        except Exception as e:
            logger.error(
                "Failed to clear user memory",
                extra={'user_id': user_id, 'error': str(e)},
                exc_info=True
            )
            raise

    async def close(self):
        """Close database connections."""
        if self.backend == 'postgres' and self._db:
            await self._db.close()
            logger.info("Semantic memory connection closed")
