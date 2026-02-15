# Unified A2A Protocol

**Version:** 1.0.0  
**Date:** February 15, 2026  
**Status:** MVP

## Overview

The Unified A2A Protocol combines the best features from major agent-to-agent communication protocols:

- **Google A2A**: Agent cards, task management, capability discovery
- **MCP (Model Context Protocol)**: Tool calling, resource access, AI assistant integration
- **AutoGen**: Multi-agent orchestration, group chats, code execution
- **LangGraph**: Workflow orchestration, state management, checkpoints
- **CrewAI**: Crew-based collaboration, tool sharing, workflow tracing

## Key Features

### Multi-Protocol Support

```python
from src.sdk.client import UnifiedClient
from src.protocols.unified.message import ProtocolType

# Create client with specific protocol
client = UnifiedClient(default_protocol=ProtocolType.A2A)

# Or let it auto-detect
client = UnifiedClient()
```

### Unified Message Format

All protocols use a common message format:

```python
from src.protocols.unified.message import UnifiedMessage, MessageType, ProtocolType

message = UnifiedMessage(
    protocol=ProtocolType.A2A,
    message_type=MessageType.TASK,
    sender_id="agent-1",
    receiver_id="agent-2",
    content={"prompt": "Process this data"},
    task_id="task-123",
)
```

### Agent Discovery

```python
from src.sdk.discovery import AgentDiscovery

async with AgentDiscovery() as discovery:
    # Discover agent
    agent = await discovery.discover("https://agent.example.com")
    
    # Search by capability
    agents = await discovery.find_by_capability(CapabilityType.TOOLS)
    
    # Health check
    healthy = await discovery.health_check(agent.id)
```

### Task Management

```python
from src.sdk.task_manager import TaskManager

manager = TaskManager()

# Create task
task = await manager.create_task(
    agent_id="agent-123",
    task_data={"prompt": "Process this"},
    priority=2,
)

# Execute with retries
result = await manager.execute_task(task.id)

# Wait for completion
completed_task = await manager.wait_for_task(task.id)

# Batch execution
tasks = await manager.create_batch([
    {"agent_id": "agent-1", "task_data": {"x": 1}},
    {"agent_id": "agent-2", "task_data": {"x": 2}},
])
```

### Workflow Orchestration

```python
from src.sdk.workflow import WorkflowEngine, Workflow, WorkflowStep

engine = WorkflowEngine()

# Define workflow
workflow = Workflow(
    id="data-pipeline",
    name="Data Processing Pipeline",
    steps={
        "fetch": WorkflowStep(
            id="fetch",
            name="Fetch Data",
            agent_id="agent-1",
            task_data={"action": "fetch"},
        ),
        "process": WorkflowStep(
            id="process",
            name="Process Data",
            agent_id="agent-2",
            task_data={"action": "process"},
            dependencies=["fetch"],
        ),
        "save": WorkflowStep(
            id="save",
            name="Save Results",
            agent_id="agent-3",
            task_data={"action": "save"},
            dependencies=["process"],
        ),
    },
)

# Execute
final_state = await engine.execute(workflow)

# Parallel execution
final_state = await engine.execute_parallel(workflow, max_concurrent=5)
```

## Protocol Adapters

### Google A2A Adapter

```python
from src.protocols.adapters import A2AAdapter

adapter = A2AAdapter(agent_card)

# Convert to unified
unified = await adapter.to_unified(a2a_message)

# Convert from unified
a2a = await adapter.from_unified(unified)
```

### MCP Adapter

```python
from src.protocols.adapters import MCPAdapter

adapter = MCPAdapter(agent_card)

# Register tools
adapter.register_tool(
    name="calculator",
    description="Perform calculations",
    input_schema={"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
    handler=lambda x, y: x + y,
)

# Register resources
adapter.register_resource(
    uri="data://config",
    name="Configuration",
    description="System configuration",
)
```

## CLI Usage

### Installation

```bash
pip install unified-a2a
```

### Commands

```bash
# Discover agent
unified-a2a discover https://agent.example.com

# Send message
unified-a2a send agent-123 "Hello!" --type text

# Create task
unified-a2a task create agent-123 --data '{"prompt": "Process this"}' --priority 2

# Check task status
unified-a2a task status task-123

# List tasks
unified-a2a task list --status running

# Execute workflow
unified-a2a workflow execute workflow.json

# Start server
unified-a2a serve --port 8000

# Protocol conversion
unified-a2a protocol convert message.json --from a2a --to mcp
```

## Security

### DID-Based Identity

```python
from src.identity.key_manager import KeyManager

# Generate keys
key_manager = KeyManager()
private_key, public_key = key_manager.generate_ed25519_keypair()

# Create DID
did = f"did:key:{key_manager.key_to_base64(public_key)[:32]}"

# Sign message
signature = key_manager.sign(private_key, message_bytes)

# Verify signature
is_valid = key_manager.verify(public_key, message_bytes, signature)
```

### Message Encryption

```python
from src.security.encryption import encrypt_message, decrypt_message

# Encrypt
encrypted = encrypt_message(
    content=message_bytes,
    recipient_public_key=public_key,
)

# Decrypt
decrypted = decrypt_message(
    encrypted=encrypted,
    recipient_private_key=private_key,
)
```

## Configuration

### Environment Variables

```bash
# Server configuration
A2A_HOST=localhost
A2A_PORT=8000

# Security
SECRET_KEY=your-secret-key
ENABLE_ENCRYPTION=true

# Protocol defaults
DEFAULT_PROTOCOL=a2a

# Logging
LOG_LEVEL=INFO
```

### Configuration File

```yaml
# config.yaml
server:
  host: localhost
  port: 8000

protocols:
  default: a2a
  enabled:
    - a2a
    - mcp
    - autogen
    - langgraph
    - crewai

security:
  enable_encryption: true
  require_signatures: true

logging:
  level: INFO
  format: json
```

## API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Support

- Documentation: https://docs.rl-a2a.io
- Issues: https://github.com/rl-a2a/unified-protocol/issues
- Discussions: https://github.com/rl-a2a/unified-protocol/discussions

---

**Built with ❤️ by the RL-A2A Team**