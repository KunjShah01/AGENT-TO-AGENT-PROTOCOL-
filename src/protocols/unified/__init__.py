"""
Unified A2A Protocol - Combining Google A2A, MCP, AutoGen, LangGraph, and CrewAI

This module provides a unified protocol specification that integrates:
- Google A2A: Agent cards, task management, capability discovery
- MCP (Model Context Protocol): Tool calling, resource access
- AutoGen: Multi-agent orchestration, group chats
- LangGraph: Workflow orchestration, state management
- CrewAI: Crew-based collaboration, tool sharing

Author: RL-A2A Team
Version: 1.0.0
"""

from .message import (
    UnifiedMessage,
    MessageType,
    MessagePriority,
    ProtocolType,
    TaskStatus,
)
from .agent_card import (
    AgentCard,
    AgentCapability,
    AgentSkill,
    AgentEndpoint,
)
from .task import (
    Task,
    TaskResult,
    TaskArtifact,
)
from .protocol import (
    UnifiedProtocol,
    ProtocolAdapter,
    ProtocolCapability,
)

__all__ = [
    # Messages
    "UnifiedMessage",
    "MessageType",
    "MessagePriority",
    "ProtocolType",
    "TaskStatus",
    # Agent Cards
    "AgentCard",
    "AgentCapability",
    "AgentSkill",
    "AgentEndpoint",
    # Tasks
    "Task",
    "TaskResult",
    "TaskArtifact",
    # Protocol
    "UnifiedProtocol",
    "ProtocolAdapter",
    "ProtocolCapability",
]

__version__ = "1.0.0"