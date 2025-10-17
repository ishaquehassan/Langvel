"""
Agent Routes

Register your agents here using Laravel-inspired syntax.
"""

from langvel.routing.router import AgentRouter
from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.code_review_agent import CodeReviewAgent
from app.agents.simpletest import SimpleTest

# Create router instance
router = AgentRouter()

# Register simple test agent (no external dependencies)
@router.flow('/test')
class TestFlow(SimpleTest):
    """Simple test agent for demonstrations."""
    pass

# Register customer support agent
@router.flow('/customer-support')
class CustomerSupportFlow(CustomerSupportAgent):
    """Customer support agent for handling inquiries."""
    pass

# Register code review agent
@router.flow('/code-review')
class CodeReviewFlow(CodeReviewAgent):
    """Code review agent for analyzing code quality."""
    pass
