"""State models - The foundation of agent state management."""

from typing import Any, Dict, List, Optional, Annotated
from pydantic import BaseModel, Field, ConfigDict
from operator import add


class StateModel(BaseModel):
    """
    Base state model for all agents.

    Similar to Laravel's Eloquent models, but for agent state management.
    Uses Pydantic for validation and LangGraph for state persistence.

    Example:
        class CustomerRequestState(StateModel):
            user_id: str
            query: str
            auth_state: str = 'guest'
            sentiment: Optional[float] = None
            response: Optional[str] = None
            messages: Annotated[List[str], operator.add] = Field(default_factory=list)

            class Config:
                checkpointer = 'postgres'
                interrupts = ['before_response']
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Common fields that most agents will use
    messages: Annotated[List[Dict[str, Any]], add] = Field(
        default_factory=list,
        description="Conversation history"
    )

    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the execution"
    )

    next_step: Optional[str] = Field(
        default=None,
        description="Next step in the workflow"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if any"
    )

    # Langvel-specific configuration (not Pydantic Config)
    # These are used by the framework for checkpointing and interrupts
    _checkpointer: str = "memory"  # memory, postgres, redis
    _interrupts: List[str] = []  # Nodes to interrupt before/after

    def update(self, **kwargs) -> "StateModel":
        """
        Update the state with new values.

        Args:
            **kwargs: Fields to update

        Returns:
            Updated state instance
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def add_message(self, role: str, content: str) -> "StateModel":
        """
        Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            Updated state instance
        """
        self.messages.append({
            "role": role,
            "content": content
        })
        return self

    def set_next(self, step: str) -> "StateModel":
        """
        Set the next step in the workflow.

        Args:
            step: Name of the next step

        Returns:
            Updated state instance
        """
        self.next_step = step
        return self

    def set_error(self, error: str) -> "StateModel":
        """
        Set an error message.

        Args:
            error: Error message

        Returns:
            Updated state instance
        """
        self.error = error
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateModel":
        """Create state from dictionary."""
        return cls(**data)


class AuthenticatedState(StateModel):
    """State model with authentication fields."""

    user_id: Optional[str] = None
    auth_state: str = Field(default="guest", description="Authentication state")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    session_id: Optional[str] = None

    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.auth_state == "authenticated" and self.user_id is not None

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions


class RAGState(StateModel):
    """State model with RAG (Retrieval Augmented Generation) fields."""

    query: str = Field(description="User query")
    retrieved_docs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved documents from RAG"
    )
    rag_context: str = Field(
        default="",
        description="Formatted context from retrieved documents"
    )
    embeddings: Optional[List[float]] = None

    def add_retrieved_doc(self, doc: Dict[str, Any]) -> "RAGState":
        """Add a retrieved document."""
        self.retrieved_docs.append(doc)
        return self

    def format_context(self) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(self.retrieved_docs, 1):
            content = doc.get('content', '')
            source = doc.get('source', 'Unknown')
            context_parts.append(f"[{i}] Source: {source}\n{content}")
        self.rag_context = "\n\n".join(context_parts)
        return self.rag_context
