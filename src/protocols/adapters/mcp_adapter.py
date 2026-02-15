"""
MCP (Model Context Protocol) Adapter

Implements Anthropic's Model Context Protocol for AI assistant integration:
- Tool calling
- Resource access
- Sampling
- Context management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..unified.message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from ..unified.agent_card import AgentCard, AgentCapability, CapabilityType
from ..unified.task import Task, TaskResult, TaskArtifact
from ..unified.protocol import ProtocolAdapter as BaseProtocolAdapter, ProtocolCapability


class MCPAdapter(BaseProtocolAdapter):
    """
    MCP Protocol Adapter
    
    Implements the Model Context Protocol for AI assistant integration.
    
    Key features:
    - Tool calling (tools/list, tools/call)
    - Resource access (resources/list, resources/read)
    - Sampling (sampling/createMessage)
    - Context management
    """
    
    def __init__(self, agent_card: Optional[AgentCard] = None):
        super().__init__(
            name="Model Context Protocol",
            protocol_type=ProtocolType.MCP,
            version="1.0.0",
            capabilities=[
                ProtocolCapability.TOOL_CALLING,
                ProtocolCapability.RESOURCE_ACCESS,
                ProtocolCapability.STREAMING,
            ],
        )
        
        self.agent_card = agent_card
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._resources: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._logger = logging.getLogger(__name__)
        
        # Register MCP methods
        self._register_methods()
    
    def _register_methods(self):
        """Register MCP protocol methods"""
        self._handlers["tools/list"] = self._handle_tools_list
        self._handlers["tools/call"] = self._handle_tools_call
        self._handlers["resources/list"] = self._handle_resources_list
        self._handlers["resources/read"] = self._handle_resources_read
        self._handlers["sampling/createMessage"] = self._handle_sampling
        self._handlers["agent/card"] = self._handle_agent_card
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], 
                     handler: Callable) -> None:
        """
        Register an MCP tool
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON schema for input validation
            handler: Tool handler function
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
        }
        self._logger.info(f"Registered MCP tool: {name}")
    
    def register_resource(self, uri: str, name: str, description: str,
                         mime_type: str = "application/json") -> None:
        """
        Register an MCP resource
        
        Args:
            uri: Resource URI
            name: Resource name
            description: Resource description
            mime_type: MIME type
        """
        self._resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description,
            "mimeType": mime_type,
        }
        self._logger.info(f"Registered MCP resource: {uri}")
    
    async def to_unified(self, mcp_message: Dict[str, Any]) -> UnifiedMessage:
        """
        Convert MCP message to unified format
        
        Args:
            mcp_message: MCP format message
            
        Returns:
            UnifiedMessage
        """
        method = mcp_message.get("method", "")
        params = mcp_message.get("params", {})
        
        # Determine message type
        message_type = MessageType.TOOL_CALL
        if "tools/list" in method:
            message_type = MessageType.QUERY
        elif "resources" in method:
            message_type = MessageType.RESOURCE_REQUEST
        elif "sampling" in method:
            message_type = MessageType.QUERY
        
        unified = UnifiedMessage(
            id=mcp_message.get("id", str(uuid.uuid4())),
            protocol=ProtocolType.MCP,
            message_type=message_type,
            sender_id=params.get("_meta", {}).get("sender", ""),
            receiver_id="",  # MCP is typically server-side
            content=params,
            tool_calls=[{"name": method, "arguments": params}] if method else [],
            metadata={
                "mcp_method": method,
                "jsonrpc_id": mcp_message.get("id"),
            }
        )
        
        return unified
    
    async def from_unified(self, unified: UnifiedMessage) -> Dict[str, Any]:
        """
        Convert unified message to MCP format
        
        Args:
            unified: UnifiedMessage
            
        Returns:
            MCP format message
        """
        # Determine method from message type
        method_map = {
            MessageType.TOOL_CALL: "tools/call",
            MessageType.TOOL_RESULT: "tools/result",
            MessageType.RESOURCE_REQUEST: "resources/read",
            MessageType.RESOURCE_RESPONSE: "resources/response",
            MessageType.QUERY: "sampling/createMessage",
            MessageType.RESPONSE: "sampling/response",
        }
        
        method = method_map.get(unified.message_type, "tools/call")
        
        mcp_message = {
            "jsonrpc": "2.0",
            "id": unified.id,
            "method": method,
            "params": {
                "content": unified.content,
                "tool_results": unified.tool_results,
                "metadata": unified.metadata,
            }
        }
        
        if unified.tool_calls:
            mcp_message["params"]["name"] = unified.tool_calls[0].get("name")
            mcp_message["params"]["arguments"] = unified.tool_calls[0].get("arguments", {})
        
        return mcp_message
    
    async def handle_message(self, message: UnifiedMessage) -> Optional[UnifiedMessage]:
        """
        Handle unified message
        
        Args:
            message: UnifiedMessage
            
        Returns:
            Response message or None
        """
        method = message.metadata.get("mcp_method", "tools/call")
        handler = self._handlers.get(method)
        
        if handler:
            return await handler(message)
        
        return None
    
    async def _handle_tools_list(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle tools/list request"""
        tools = [
            {
                "name": name,
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            }
            for name, tool in self._tools.items()
        ]
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.MCP,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={"tools": tools},
            correlation_id=message.id,
        )
    
    async def _handle_tools_call(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle tools/call request"""
        tool_name = message.tool_calls[0].get("name") if message.tool_calls else None
        arguments = message.tool_calls[0].get("arguments", {}) if message.tool_calls else message.content
        
        if not tool_name or tool_name not in self._tools:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.MCP,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Tool not found: {tool_name}"},
                correlation_id=message.id,
            )
        
        # Execute tool
        try:
            tool = self._tools[tool_name]
            handler = tool["handler"]
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)
            
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.MCP,
                message_type=MessageType.TOOL_RESULT,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content=result,
                tool_results=[{"name": tool_name, "result": result}],
                correlation_id=message.id,
            )
            
        except Exception as e:
            self._logger.error(f"Tool execution failed: {e}")
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.MCP,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": str(e), "tool": tool_name},
                correlation_id=message.id,
            )
    
    async def _handle_resources_list(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle resources/list request"""
        resources = [
            {
                "uri": uri,
                "name": resource["name"],
                "description": resource["description"],
                "mimeType": resource["mimeType"],
            }
            for uri, resource in self._resources.items()
        ]
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.MCP,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={"resources": resources},
            correlation_id=message.id,
        )
    
    async def _handle_resources_read(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle resources/read request"""
        uri = message.content.get("uri") if isinstance(message.content, dict) else None
        
        if not uri or uri not in self._resources:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.MCP,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Resource not found: {uri}"},
                correlation_id=message.id,
            )
        
        resource = self._resources[uri]
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.MCP,
            message_type=MessageType.RESOURCE_RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "uri": uri,
                "name": resource["name"],
                "description": resource["description"],
                "mimeType": resource["mimeType"],
            },
            correlation_id=message.id,
        )
    
    async def _handle_sampling(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle sampling/createMessage request"""
        # This would typically call an LLM
        prompt = message.content.get("prompt") if isinstance(message.content, dict) else str(message.content)
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.MCP,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "content": f"Sampled response for: {prompt[:50]}...",
                "model": "unified-sampler",
                "stop_reason": "end_turn",
            },
            correlation_id=message.id,
        )
    
    async def handle_task(self, task: Task) -> TaskResult:
        """
        Execute task using MCP tools
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult
        """
        try:
            # Check if task requires tool execution
            if task.input_data and isinstance(task.input_data, dict):
                tool_name = task.input_data.get("tool")
                if tool_name and tool_name in self._tools:
                    # Execute tool
                    tool = self._tools[tool_name]
                    handler = tool["handler"]
                    arguments = task.input_data.get("arguments", {})
                    
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(**arguments)
                    else:
                        result = handler(**arguments)
                    
                    return TaskResult(
                        status=TaskStatus.COMPLETED,
                        output=result,
                        execution_time=0.1,
                    )
            
            # Default: return success
            return TaskResult(
                status=TaskStatus.COMPLETED,
                output={"message": "Task processed via MCP"},
            )
            
        except Exception as e:
            self._logger.error(f"MCP task execution failed: {e}")
            return TaskResult(
                status=TaskStatus.FAILED,
                error=str(e),
                error_code="MCP_ERROR",
            )


