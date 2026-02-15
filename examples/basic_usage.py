"""
Basic Usage Example

Demonstrates basic usage of the Unified A2A SDK.
"""

import asyncio
from src.sdk.client import UnifiedClient
from src.sdk.agent import AgentClient
from src.sdk.discovery import AgentDiscovery
from src.protocols.unified.message import MessageType, ProtocolType


async def main():
    """Main example"""
    print("=" * 60)
    print("Unified A2A SDK - Basic Usage Example")
    print("=" * 60)
    
    # Example 1: Create client and discover agent
    print("\n1. Agent Discovery")
    print("-" * 40)
    
    async with AgentDiscovery() as discovery:
        # Discover agent at endpoint
        agent_card = await discovery.discover("http://localhost:8000")
        
        if agent_card:
            print(f"Discovered: {agent_card.name}")
            print(f"ID: {agent_card.id}")
            print(f"Capabilities: {[c.type.value for c in agent_card.capabilities]}")
        else:
            print("No agent found (this is expected if no server is running)")
    
    # Example 2: Send message
    print("\n2. Send Message")
    print("-" * 40)
    
    async with UnifiedClient() as client:
        # This would work if an agent is running
        try:
            response = await client.send_message(
                agent_id="agent-123",
                content={"message": "Hello, Agent!"},
                message_type=MessageType.TEXT,
            )
            print(f"Response: {response.content}")
        except Exception as e:
            print(f"Message sending failed (expected if no agent): {e}")
    
    # Example 3: Create and execute task
    print("\n3. Task Management")
    print("-" * 40)
    
    from src.sdk.task_manager import TaskManager
    
    manager = TaskManager()
    
    # Create task
    task = await manager.create_task(
        agent_id="agent-123",
        task_data={
            "name": "Example Task",
            "description": "Process some data",
            "data": {"key": "value"},
        },
        priority=2,
    )
    
    print(f"Created task: {task.id}")
    print(f"Status: {task.status.value}")
    print(f"Priority: {task.priority.value}")
    
    # Execute task
    result = await manager.execute_task(task.id)
    
    print(f"\nExecution result:")
    print(f"Status: {result.status.value}")
    if result.is_success():
        print(f"Output: {result.output}")
    else:
        print(f"Error: {result.error}")
    
    # Example 4: Workflow
    print("\n4. Workflow Execution")
    print("-" * 40)
    
    from src.sdk.workflow import WorkflowEngine, Workflow, WorkflowStep
    
    engine = WorkflowEngine()
    
    # Create workflow
    workflow = Workflow(
        id="example-workflow",
        name="Data Processing Pipeline",
        steps={
            "fetch": WorkflowStep(
                id="fetch",
                name="Fetch Data",
                agent_id="agent-1",
                task_data={"action": "fetch_data"},
            ),
            "process": WorkflowStep(
                id="process",
                name="Process Data",
                agent_id="agent-2",
                task_data={"action": "process_data"},
                dependencies=["fetch"],
            ),
            "save": WorkflowStep(
                id="save",
                name="Save Results",
                agent_id="agent-3",
                task_data={"action": "save_results"},
                dependencies=["process"],
            ),
        },
        initial_state={"source": "example"},
    )
    
    print(f"Created workflow: {workflow.name}")
    print(f"Steps: {list(workflow.steps.keys())}")
    print(f"Execution order: {workflow.get_execution_order()}")
    
    # Execute workflow
    print("\nExecuting workflow...")
    final_state = await engine.execute(workflow)
    
    print("\nFinal state:")
    print_json = json.dumps(final_state, indent=2, default=str)
    print(print_json)
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


