"""
Unified Message Model
Combines message formats from Google A2A, MCP, AutoGen, LangGraph, and CrewAI
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class MessageType(str, Enum):
    """Unified message types combining all protocols"""
    # Google A2A types
    TASK = "task"
    TASK_STATUS = "task_status"
    TASK_CANCEL = "task_cancel"
    TASK_RESULT = "task_result"
    
    # MCP types
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    
    # AutoGen types
    CHAT = "chat"
    GROUP_CHAT = "group_chat"
    ORCHESTRATION = "orchestration"
    
    # LangGraph types
    STATE_UPDATE = "state_update"
    WORKFLOW_EVENT = "workflow_event"
    CHECKPOINT = "checkpoint"
    
    # CrewAI types
    CREW_TASK = "crew_task"
    TOOL_SHARE = "tool_share"
    COLLABORATION = "collaboration"
    
    # General types
    TEXT = "text"
    NOTIFICATION = "notification"
    QUERY = "query"
    COMMAND = "command"
    RESPONSE = "response"
    ERROR = "error"


class MessagePriority(int, Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class ProtocolType(str, Enum):
    """Protocol types for routing"""
    A2A = "a2a"           # Google A2A
    MCP = "mcp"           # Model Context Protocol
    AUTOGEN = "autogen"   # Microsoft AutoGen
    LANGGRAPH = "langgraph" # LangGraph
    CREWAI = "crewai"     # CrewAI
    INTERNAL = "internal" # Internal protocol
    REST = "rest"         # REST API
    WEBSOCKET = "websocket" # WebSocket
    GRPC = "grpc"         # gRPC


class TaskStatus(str, Enum):
    """Task status enumeration (Google A2A compatible)"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class UnifiedMessage:
    """
    Unified Message Model
    
    Combines features from:
    - Google A2A: Task management, agent cards
    - MCP: Tool calling, resource access
    - AutoGen: Group chat, orchestration
    - LangGraph: State management, workflows
    - CrewAI: Crew tasks, collaboration
    
    Attributes:
        id: Unique message identifier
        protocol: Source protocol type
        message_type: Type of message
        
        # Identifiers
        sender_id: Sender agent ID
        sender_did: Sender DID (decentralized identifier)
        receiver_id: Receiver agent ID
        receiver_did: Receiver DID
        
        # Content
        content: Message content/payload
        content_type: MIME type of content
        
        # Task Management (A2A)
        task_id: Task identifier
        task_status: Task status
        task_result: Task result data
        
        # Tool Calling (MCP)
        tool_calls: List of tool calls
        tool_results: List of tool results
        
        # State Management (LangGraph)
        state: Workflow state
        checkpoint_id: Checkpoint identifier
        
        # Collaboration (CrewAI)
        crew_id: Crew identifier
        collaboration_context: Shared context
        
        # Metadata
        priority: Message priority
        timestamp: Message timestamp
        correlation_id: Correlation ID for threading
        metadata: Additional metadata
        
        # Security
        encrypted: Whether message is encrypted
        signature: Message signature
        requires_approval: Whether HITL approval is needed
    """
    
    # Core identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protocol: ProtocolType = ProtocolType.INTERNAL
    message_type: MessageType = MessageType.TEXT
    
    # Agent identifiers
    sender_id: str = ""
    sender_did: Optional[str] = None
    receiver_id: str = ""
    receiver_did: Optional[str] = None
    
    # Content
    content: Any = None
    content_type: str = "application/json"
    
    # Task Management (A2A)
    task_id: Optional[str] = None
    task_status: Optional[TaskStatus] = None
    task_result: Optional[Dict[str, Any]] = None
    
    # Tool Calling (MCP)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # State Management (LangGraph)
    state: Optional[Dict[str, Any]] = None
    checkpoint_id: Optional[str] = None
    
    # Collaboration (CrewAI)
    crew_id: Optional[str] = None
    collaboration_context: Optional[Dict[str, Any]] = None
    
    # Metadata
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Security
    encrypted: bool = False
    signature: Optional[str] = None
    requires_approval: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "protocol": self.protocol.value,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "sender_did": self.sender_did,
            "receiver_id": self.receiver_id,
            "receiver_did": self.receiver_did,
            "content": self.content,
            "content_type": self.content_type,
            "task_id": self.task_id,
            "task_status": self.task_status.value if self.task_status else None,
            "task_result": self.task_result,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "state": self.state,
            "checkpoint_id": self.checkpoint_id,
            "crew_id": self.crew_id,
            "collaboration_context": self.collaboration_context,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "encrypted": self.encrypted,
            "signature": self.signature,
            "requires_approval": self.requires_approval,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create message from dictionary"""
        data = data.copy()
        
        # Convert enums
        if "protocol" in data:
            data["protocol"] = ProtocolType(data["protocol"])
        if "message_type" in data:
            data["message_type"] = MessageType(data["message_type"])
        if "task_status" in data and data["task_status"]:
            data["task_status"] = TaskStatus(data["task_status"])
        if "priority" in data:
            data["priority"] = MessagePriority(data["priority"])
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return cls(**data)
    
    def to_jsonrpc(self) -> Dict[str, Any]:
        """Convert to JSON-RPC 2.0 format"""
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": f"{self.protocol.value}/{self.message_type.value}",
            "params": {
                "sender_id": self.sender_id,
                "sender_did": self.sender_did,
                "receiver_id": self.receiver_id,
                "receiver_did": self.receiver_did,
                "content": self.content,
                "task_id": self.task_id,
                "tool_calls": self.tool_calls,
                "state": self.state,
                "priority": self.priority.value,
                "metadata": self.metadata,
            }
        }
    
    def to_a2a(self) -> Dict[str, Any]:
        """Convert to Google A2A format"""
        return {
            "id": self.id,
            "sender": self.sender_id,
            "receiver": self.receiver_id,
            "type": self.message_type.value,
            "content": self.content,
            "task_id": self.task_id,
            "task_status": self.task_status.value if self.task_status else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def to_mcp(self) -> Dict[str, Any]:
        """Convert to MCP format"""
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": "tools/call" if self.tool_calls else "resources/read",
            "params": {
                "name": self.metadata.get("tool_name"),
                "arguments": self.content,
                "tool_calls": self.tool_calls,
            }
        }
    
    def to_autogen(self) -> Dict[str, Any]:
        """Convert to AutoGen format"""
        return {
            "content": self.content,
            "role": "assistant" if self.sender_id else "user",
            "name": self.sender_id,
            "function_call": self.tool_calls[0] if self.tool_calls else None,
            "metadata": {
                "group_id": self.crew_id,
                "checkpoint_id": self.checkpoint_id,
                **self.metadata
            }
        }
    
    def to_crewai(self) -> Dict[str, Any]:
        """Convert to CrewAI format"""
        return {
            "description": self.content,
            "agent": self.sender_id,
            "context": self.collaboration_context,
            "tools": self.tool_calls,
            "crew": self.crew_id,
            "priority": self.priority.value,
            "async_execution": self.metadata.get("async", False),
        }
    
    def to_langgraph(self) -> Dict[str, Any]:
        """Convert to LangGraph format"""
        return {
            "messages": [{"role": "user", "content": self.content}],
            "state": self.state,
            "checkpoint": self.checkpoint_id,
            "metadata": {
                "sender": self.sender_id,
                "receiver": self.receiver_id,
                "task_id": self.task_id,
                **self.metadata
            }
        }
