"""
Unified Agent Card Specification
Combines Google A2A Agent Cards, MCP capabilities, and DID-based identity
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class CapabilityType(str, Enum):
    """Agent capability types"""
    # Google A2A capabilities
    TASK_EXECUTION = "task_execution"
    STREAMING = "streaming"
    PUSH_NOTIFICATIONS = "push_notifications"
    
    # MCP capabilities
    TOOLS = "tools"
    RESOURCES = "resources"
    SAMPLING = "sampling"
    
    # AutoGen capabilities
    GROUP_CHAT = "group_chat"
    CODE_EXECUTION = "code_execution"
    
    # LangGraph capabilities
    WORKFLOW = "workflow"
    STATE_MANAGEMENT = "state_management"
    
    # CrewAI capabilities
    COLLABORATION = "collaboration"
    TOOL_SHARING = "tool_sharing"
    
    # General capabilities
    COMMUNICATION = "communication"
    LEARNING = "learning"
    REASONING = "reasoning"
    MEMORY = "memory"


class SkillCategory(str, Enum):
    """Skill categories for agent capabilities"""
    CONVERSATIONAL = "conversational"
    ANALYTICS = "analytics"
    DEVELOPMENT = "development"
    SECURITY = "security"
    OPTIMIZATION = "optimization"
    RESEARCH = "research"
    COORDINATION = "coordination"
    GENERAL = "general"


@dataclass
class AgentSkill:
    """
    Agent skill definition
    
    Combines Google A2A skill definitions with MCP tool descriptions
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: SkillCategory = SkillCategory.GENERAL
    
    # Input/output schemas (JSON Schema format)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Examples for few-shot prompting
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "examples": self.examples,
            "tags": self.tags,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AgentCapability:
    """
    Agent capability declaration
    
    Combines Google A2A capabilities with MCP protocol features
    """
    type: CapabilityType
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "enabled": self.enabled,
            "config": self.config,
        }


@dataclass
class AgentEndpoint:
    """
    Agent endpoint configuration
    
    Defines how to communicate with an agent
    """
    protocol: ProtocolType
    url: str
    authentication: Optional[Dict[str, Any]] = None
    capabilities: List[CapabilityType] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol.value,
            "url": self.url,
            "authentication": self.authentication,
            "capabilities": [c.value for c in self.capabilities],
        }


@dataclass
class AgentCard:
    """
    Unified Agent Card
    
    Combines:
    - Google A2A Agent Card format
    - MCP server capabilities
    - DID-based identity
    - AutoGen/LangGraph/CrewAI features
    
    This is the primary discovery mechanism for agents.
    """
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    did: Optional[str] = None
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Display
    display_name: Optional[str] = None
    icon_url: Optional[str] = None
    
    # Capabilities (combined from all protocols)
    capabilities: List[AgentCapability] = field(default_factory=list)
    
    # Skills (for capability discovery)
    skills: List[AgentSkill] = field(default_factory=list)
    
    # Endpoints (how to communicate)
    endpoints: List[AgentEndpoint] = field(default_factory=list)
    
    # Authentication
    authentication: Optional[Dict[str, Any]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Documentation
    documentation_url: Optional[str] = None
    terms_of_service: Optional[str] = None
    privacy_policy: Optional[str] = None
    
    # Contact
    contact_email: Optional[str] = None
    contact_url: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize default capabilities if none provided"""
        if not self.capabilities:
            self.capabilities = [
                AgentCapability(CapabilityType.COMMUNICATION),
                AgentCapability(CapabilityType.TASK_EXECUTION),
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (Google A2A compatible)"""
        return {
            "id": self.id,
            "did": self.did,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "display_name": self.display_name,
            "icon_url": self.icon_url,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "skills": [s.to_dict() for s in self.skills],
            "endpoints": [e.to_dict() for e in self.endpoints],
            "authentication": self.authentication,
            "tags": self.tags,
            "categories": self.categories,
            "documentation_url": self.documentation_url,
            "terms_of_service": self.terms_of_service,
            "privacy_policy": self.privacy_policy,
            "contact_email": self.contact_email,
            "contact_url": self.contact_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Create from dictionary"""
        # Parse capabilities
        capabilities = []
        for cap_data in data.get("capabilities", []):
            capabilities.append(AgentCapability(
                type=CapabilityType(cap_data["type"]),
                enabled=cap_data.get("enabled", True),
                config=cap_data.get("config", {})
            ))
        
        # Parse skills
        skills = []
        for skill_data in data.get("skills", []):
            skills.append(AgentSkill(
                id=skill_data.get("id", str(uuid.uuid4())),
                name=skill_data["name"],
                description=skill_data.get("description", ""),
                category=SkillCategory(skill_data.get("category", "general")),
                input_schema=skill_data.get("input_schema", {}),
                output_schema=skill_data.get("output_schema", {}),
                examples=skill_data.get("examples", []),
                tags=skill_data.get("tags", []),
                version=skill_data.get("version", "1.0.0"),
            ))
        
        # Parse endpoints
        endpoints = []
        for ep_data in data.get("endpoints", []):
            endpoints.append(AgentEndpoint(
                protocol=ProtocolType(ep_data["protocol"]),
                url=ep_data["url"],
                authentication=ep_data.get("authentication"),
                capabilities=[CapabilityType(c) for c in ep_data.get("capabilities", [])]
            ))
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            did=data.get("did"),
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            display_name=data.get("display_name"),
            icon_url=data.get("icon_url"),
            capabilities=capabilities,
            skills=skills,
            endpoints=endpoints,
            authentication=data.get("authentication"),
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            documentation_url=data.get("documentation_url"),
            terms_of_service=data.get("terms_of_service"),
            privacy_policy=data.get("privacy_policy"),
            contact_email=data.get("contact_email"),
            contact_url=data.get("contact_url"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


