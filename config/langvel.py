"""
Langvel Configuration

Central configuration file for Langvel framework.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Main configuration class."""

    # LLM Configuration
    LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'anthropic')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'claude-3-5-sonnet-20241022')
    LLM_TEMPERATURE: float = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    LLM_MAX_TOKENS: int = int(os.getenv('LLM_MAX_TOKENS', '4096'))

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')

    # RAG Configuration
    RAG_PROVIDER: str = os.getenv('RAG_PROVIDER', 'chroma')
    RAG_EMBEDDING_MODEL: str = os.getenv('RAG_EMBEDDING_MODEL', 'openai/text-embedding-3-small')
    RAG_PERSIST_DIRECTORY: str = os.getenv('RAG_PERSIST_DIRECTORY', './storage/chroma_db')

    # MCP Servers Configuration
    MCP_SERVERS: Dict[str, Dict[str, Any]] = {
        # Example configurations:
        # 'slack': {
        #     'command': 'npx',
        #     'args': ['-y', '@modelcontextprotocol/server-slack'],
        #     'env': {'SLACK_BOT_TOKEN': os.getenv('SLACK_BOT_TOKEN', '')}
        # },
        # 'github': {
        #     'command': 'npx',
        #     'args': ['-y', '@modelcontextprotocol/server-github'],
        #     'env': {'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN', '')}
        # }
    }

    # State Management
    STATE_CHECKPOINTER: str = os.getenv('STATE_CHECKPOINTER', 'memory')  # memory, postgres, redis

    # Database Configuration (for postgres checkpointer)
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://localhost/langvel')
    DATABASE_POOL_SIZE: int = int(os.getenv('DATABASE_POOL_SIZE', '10'))

    # Redis Configuration (for redis checkpointer)
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))

    # Server Configuration
    SERVER_HOST: str = os.getenv('SERVER_HOST', '0.0.0.0')
    SERVER_PORT: int = int(os.getenv('SERVER_PORT', '8000'))
    SERVER_WORKERS: int = int(os.getenv('SERVER_WORKERS', '1'))

    # Middleware Configuration
    DEFAULT_MIDDLEWARE: list = ['logging']  # Applied to all agents by default

    # Security
    CORS_ORIGINS: list = os.getenv('CORS_ORIGINS', '*').split(',')
    RATE_LIMIT_REQUESTS: int = int(os.getenv('RATE_LIMIT_REQUESTS', '10'))
    RATE_LIMIT_WINDOW: int = int(os.getenv('RATE_LIMIT_WINDOW', '60'))

    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', './storage/logs/langvel.log')

    # Development
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    RELOAD: bool = os.getenv('RELOAD', 'False').lower() == 'true'


# Export singleton config instance
config = Config()
