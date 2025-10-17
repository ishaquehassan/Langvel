"""LLM Manager - Unified interface for LLM providers."""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod


class LLMManager:
    """
    Manages LLM operations across different providers.

    Supports Anthropic (Claude), OpenAI (GPT), and custom providers.
    """

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None, **kwargs):
        """
        Initialize LLM manager.

        Args:
            provider: LLM provider (anthropic, openai, custom)
            model: Model identifier
            **kwargs: Additional configuration
        """
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        self._client = None

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            from config.langvel import config

            self._client = ChatAnthropic(
                model=self.model or config.LLM_MODEL,
                temperature=self.kwargs.get('temperature', config.LLM_TEMPERATURE),
                max_tokens=self.kwargs.get('max_tokens', config.LLM_MAX_TOKENS),
                **{k: v for k, v in self.kwargs.items() if k not in ['temperature', 'max_tokens']}
            )

        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            from config.langvel import config

            self._client = ChatOpenAI(
                model=self.model or "gpt-4",
                temperature=self.kwargs.get('temperature', 0.7),
                **{k: v for k, v in self.kwargs.items() if k != 'temperature'}
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        return self._client

    async def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Invoke LLM with a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            LLM response text
        """
        client = self._get_client()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Invoke
        response = await client.ainvoke(messages, **kwargs)

        return response.content

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Stream LLM response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Response chunks
        """
        client = self._get_client()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Stream
        async for chunk in client.astream(messages, **kwargs):
            yield chunk.content

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Multi-turn conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            LLM response text
        """
        client = self._get_client()
        response = await client.ainvoke(messages, **kwargs)
        return response.content

    def with_structured_output(self, schema: type):
        """
        Get LLM client with structured output.

        Args:
            schema: Pydantic model class

        Returns:
            LLM client configured for structured output
        """
        client = self._get_client()
        return client.with_structured_output(schema)


class LLMConfig:
    """Configuration for LLM setup."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize LLM configuration.

        Args:
            provider: LLM provider
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional configuration
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    def create_manager(self) -> LLMManager:
        """
        Create LLM manager from configuration.

        Returns:
            Configured LLMManager instance
        """
        return LLMManager(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs
        )


# Convenience functions
async def ask_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick LLM query function.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        provider: LLM provider
        model: Model identifier
        **kwargs: Additional parameters

    Returns:
        LLM response

    Example:
        response = await ask_llm("What is Python?")
        response = await ask_llm(
            "Review this code",
            system_prompt="You are a code reviewer"
        )
    """
    manager = LLMManager(provider=provider, model=model, **kwargs)
    return await manager.invoke(prompt, system_prompt=system_prompt)


async def stream_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    **kwargs
):
    """
    Stream LLM response.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        provider: LLM provider
        model: Model identifier
        **kwargs: Additional parameters

    Yields:
        Response chunks

    Example:
        async for chunk in stream_llm("Tell me a story"):
            print(chunk, end="")
    """
    manager = LLMManager(provider=provider, model=model, **kwargs)
    async for chunk in manager.stream(prompt, system_prompt=system_prompt):
        yield chunk
