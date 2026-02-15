"""
Unified Task Model
Combines task management from Google A2A, LangGraph, and CrewAI
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVAL_REQUIRED = "approval_required"
    PAUSED = "paused"
    RETRYING = "retrying"


class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class TaskArtifact:
    """
    Task artifact (output file, document, etc.)
    
    Combines A2A artifact format with MCP resource format
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    mime_type: str = "application/octet-stream"
    
    # Content (one of these should be set)
    content: Optional[Union[str, bytes]] = None
    content_url: Optional[str] = None
    resource_uri: Optional[str] = None  # MCP-style resource URI
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "content": self.content if isinstance(self.content, str) else None,
            "content_url": self.content_url,
            "resource_uri": self.resource_uri,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TaskResult:
    """
    Task execution result
    
    Combines A2A result format with LangGraph state updates
    """
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    # Artifacts produced
    artifacts: List[TaskArtifact] = field(default_factory=list)
    
    # State updates (LangGraph-style)
    state_updates: Dict[str, Any] = field(default_factory=dict)
    checkpoint_id: Optional[str] = None
    
    # Metadata
    execution_time: Optional[float] = None  # seconds
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "error_code": self.error_code,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "state_updates": self.state_updates,
            "checkpoint_id": self.checkpoint_id,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "metadata": self.metadata,
            "completed_at": self.completed_at.isoformat(),
        }
    
    def is_success(self) -> bool:
        """Check if task succeeded"""
        return self.status == TaskStatus.COMPLETED
    
    def is_failure(self) -> bool:
        """Check if task failed"""
        return self.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]


@dataclass
class Task:
    """
    Unified Task Model
    
    Combines:
    - Google A2A: Task lifecycle, artifacts
    - LangGraph: State management, checkpoints
    - CrewAI: Crew assignments, collaboration
    - AutoGen: Group chat context
    """
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Assignment
    creator_id: Optional[str] = None
    assignee_id: Optional[str] = None
    crew_id: Optional[str] = None  # CrewAI-style crew
    participant_ids: List[str] = field(default_factory=list)  # AutoGen-style group
    
    # Content
    input_data: Any = None
    input_schema: Optional[Dict[str, Any]] = None
    
    # Execution
    max_steps: Optional[int] = None
    timeout: Optional[float] = None  # seconds
    retry_count: int = 0
    max_retries: int = 3
    
    # State Management (LangGraph-style)
    state: Dict[str, Any] = field(default_factory=dict)
    checkpoint_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None
    
    # Results
    result: Optional[TaskResult] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context (CrewAI-style)
    context: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "creator_id": self.creator_id,
            "assignee_id": self.assignee_id,
            "crew_id": self.crew_id,
            "participant_ids": self.participant_ids,
            "input_data": self.input_data,
            "input_schema": self.input_schema,
            "max_steps": self.max_steps,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "state": self.state,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "result": self.result.to_dict() if self.result else None,
            "history": self.history,
            "context": self.context,
            "shared_memory": self.shared_memory,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat(),
        }
    
    def update_status(self, status: TaskStatus, result: Optional[TaskResult] = None):
        """Update task status"""
        self.status = status
        self.updated_at = datetime.now()
        
        if status == TaskStatus.RUNNING and not self.started_at:
            self.started_at = datetime.now()
        
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.completed_at = datetime.now()
            if result:
                self.result = result
        
        # Add to history
        self.history.append({
            "status": status.value,
            "timestamp": self.updated_at.isoformat(),
            "result": result.to_dict() if result else None,
        })
    
    def is_active(self) -> bool:
        """Check if task is active"""
        return self.status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.RETRYING]
    
    def is_complete(self) -> bool:
        """Check if task is complete"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]