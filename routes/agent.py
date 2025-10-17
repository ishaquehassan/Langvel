"""
Agent Routes

Register your agents here using Laravel-inspired syntax.
"""

from langvel.routing.router import AgentRouter

# Create router instance
router = AgentRouter()

# Example route registration:
#
# @router.flow('/customer-support', middleware=['logging', 'auth'])
# class CustomerSupportFlow(CustomerSupportAgent):
#     """Customer support agent flow."""
#     pass
#
# # Using route groups:
# with router.group(prefix='/admin', middleware=['auth', 'admin']):
#     @router.flow('/dashboard')
#     class AdminDashboard(Agent):
#         pass
#
#     @router.flow('/reports')
#     class AdminReports(Agent):
#         pass

# Your routes go here...
