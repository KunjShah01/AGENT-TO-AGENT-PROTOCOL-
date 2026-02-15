"""
Unified A2A SDK

Python SDK for the Unified A2A Protocol.

Features:
- Multi-protocol support (A2A, MCP, AutoGen, LangGraph, CrewAI)
- Async/await support
- Agent discovery and capability negotiation
- Task management
- Workflow orchestration
- Human-in-the-loop integration
- Security (DID, signatures)

Example:
    >>> from unified_a2a import UnifiedClient, AgentCard
    >>> 
    >>> # Create client
    >>> client = UnifiedClient()
    >>> 
    >>> # Discover agent
    >>> agent = await client.discover_agent("https://agent.example.com")
    >>> 
    >>> # Send task
    >>> result = await client.send_task(
    ...     agent_id=agent.id,
    ...     task={"prompt": "Hello, agent!"}
    ... )
"""

from .client import UnifiedClient
from .agent import AgentClient
from .discovery import AgentDiscovery
from .task_manager import TaskManager
from .workflow import WorkflowEngine

__version__ = "1.0.0"

__all__ = [
    "UnifiedClient",
    "AgentClient",
    "AgentDiscovery",
    "TaskManager",
    "WorkflowEngine",
]