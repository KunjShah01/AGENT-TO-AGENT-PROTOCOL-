"""
Unified A2A CLI

Command-line interface for the Unified A2A Protocol.

Commands:
    discover    Discover agents at endpoints
    send        Send message to agent
    task        Create and manage tasks
    workflow    Execute workflows
    chat        Interactive chat with agent
    serve       Start A2A server
    protocol    Protocol utilities

Example:
    $ unified-a2a discover https://agent.example.com
    $ unified-a2a send agent-123 "Hello!"
    $ unified-a2a task create agent-123 --data '{"prompt": "Process this"}'
    $ unified-a2a serve --port 8000
"""

import asyncio
import json
import sys
from typing import Optional
from pathlib import Path

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("CLI dependencies not installed. Install with:")
    print("pip install click rich")
    sys.exit(1)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sdk.client import UnifiedClient
from src.sdk.agent import AgentClient
from src.sdk.discovery import AgentDiscovery
from src.sdk.task_manager import TaskManager
from src.sdk.workflow import WorkflowEngine, Workflow, WorkflowStep
from src.protocols.unified.message import MessageType, ProtocolType, TaskStatus
from src.protocols.unified.agent_card import AgentCard, CapabilityType


console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="unified-a2a")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """Unified A2A Protocol CLI"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("url")
@click.option("--format", "output_format", type=click.Choice(["json", "table"]), default="table")
@click.pass_context
def discover(ctx, url, output_format):
    """Discover agent at URL"""
    async def _discover():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Discovering agent...", total=None)
            
            discovery = AgentDiscovery()
            agent_card = await discovery.discover(url)
            
            progress.update(task, completed=True)
        
        if not agent_card:
            console.print(f"[red]No agent found at {url}[/red]")
            return
        
        if output_format == "json":
            console.print_json(json.dumps(agent_card.to_dict(), indent=2))
        else:
            # Table format
            table = Table(title=f"Agent: {agent_card.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("ID", agent_card.id)
            table.add_row("Name", agent_card.name)
            table.add_row("Description", agent_card.description)
            table.add_row("Version", agent_card.version)
            table.add_row("DID", agent_card.did or "N/A")
            
            # Capabilities
            caps = ", ".join([c.type.value for c in agent_card.capabilities if c.enabled])
            table.add_row("Capabilities", caps or "None")
            
            # Skills
            skills = ", ".join([s.name for s in agent_card.skills])
            table.add_row("Skills", skills or "None")
            
            console.print(table)
    
    asyncio.run(_discover())


@cli.command()
@click.argument("agent_id")
@click.argument("content")
@click.option("--type", "message_type", default="text")
@click.option("--protocol", type=click.Choice(["a2a", "mcp", "autogen", "langgraph", "crewai"]))
@click.pass_context
def send(ctx, agent_id, content, message_type, protocol):
    """Send message to agent"""
    async def _send():
        # Parse content
        try:
            content_data = json.loads(content)
        except json.JSONDecodeError:
            content_data = content
        
        # Determine protocol
        protocol_type = None
        if protocol:
            protocol_type = ProtocolType(protocol)
        
        # Create client
        client = UnifiedClient(default_protocol=protocol_type or ProtocolType.A2A)
        
        with console.status("Sending message..."):
            try:
                response = await client.send_message(
                    agent_id=agent_id,
                    content=content_data,
                    message_type=MessageType(message_type),
                    protocol=protocol_type,
                )
                
                console.print("[green]Message sent successfully![/green]")
                console.print_json(json.dumps(response.to_dict(), indent=2))
                
            except Exception as e:
                console.print(f"[red]Failed to send message: {e}[/red]")
    
    asyncio.run(_send())


@cli.group()
def task():
    """Task management commands"""
    pass


@task.command("create")
@click.argument("agent_id")
@click.option("--data", "task_data", required=True, help="Task data as JSON")
@click.option("--priority", type=int, default=2, help="Task priority (1-5)")
@click.pass_context
def task_create(ctx, agent_id, task_data, priority):
    """Create new task"""
    async def _create():
        try:
            data = json.loads(task_data)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON: {e}[/red]")
            return
        
        manager = TaskManager()
        
        with console.status("Creating task..."):
            task = await manager.create_task(
                agent_id=agent_id,
                task_data=data,
                priority=priority,
            )
        
        console.print(f"[green]Task created: {task.id}[/green]")
        console.print_json(json.dumps(task.to_dict(), indent=2, default=str))
    
    asyncio.run(_create())


@task.command("status")
@click.argument("task_id")
@click.pass_context
def task_status(ctx, task_id):
    """Get task status"""
    async def _status():
        manager = TaskManager()
        task = manager.get_task(task_id)
        
        if not task:
            console.print(f"[red]Task not found: {task_id}[/red]")
            return
        
        console.print(f"[cyan]Task: {task.name}[/cyan]")
        console.print(f"Status: {task.status.value}")
        console.print(f"Priority: {task.priority.value}")
        console.print(f"Created: {task.created_at}")
        
        if task.result:
            console.print("\n[green]Result:[/green]")
            console.print_json(json.dumps(task.result.to_dict(), indent=2, default=str))
    
    asyncio.run(_status())


@task.command("list")
@click.option("--status", "filter_status", help="Filter by status")
@click.option("--agent", "agent_id", help="Filter by agent")
@click.pass_context
def task_list(ctx, filter_status, agent_id):
    """List tasks"""
    async def _list():
        manager = TaskManager()
        
        status_enum = None
        if filter_status:
            try:
                status_enum = TaskStatus(filter_status)
            except ValueError:
                console.print(f"[red]Invalid status: {filter_status}[/red]")
                return
        
        tasks = manager.list_tasks(status=status_enum, agent_id=agent_id)
        
        if not tasks:
            console.print("No tasks found")
            return
        
        table = Table(title="Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Agent", style="yellow")
        table.add_column("Created", style="blue")
        
        for task in tasks:
            table.add_row(
                task.id[:8],
                task.name,
                task.status.value,
                task.assignee_id[:8] if task.assignee_id else "N/A",
                task.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        
        console.print(table)
        console.print(f"\nTotal: {len(tasks)} tasks")
    
    asyncio.run(_list())


@cli.command()
@click.option("--port", default=8000, help="Server port")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--protocol", type=click.Choice(["a2a", "mcp", "all"]), default="all")
@click.pass_context
def serve(ctx, port, host, protocol):
        """Start A2A server"""
        async def _serve():
            console.print(Panel.fit(f"Starting Unified A2A Server\nProtocol: {protocol}\nHost: {host}:{port}"))
            
            # This would start the actual server
            # For now, just print instructions
            console.print("\n[yellow]Server implementation would start here[/yellow]")
            console.print("To implement, use:")
            console.print("  - FastAPI for REST API")
            console.print("  - WebSocket for real-time communication")
            console.print("  - Protocol adapters for multi-protocol support")
        
        asyncio.run(_serve())


@cli.group()
def protocol():
    """Protocol utilities"""
    pass


@protocol.command("convert")
@click.argument("input_file")
@click.option("--from", "from_protocol", required=True, type=click.Choice(["a2a", "mcp", "autogen", "langgraph", "crewai"]))
@click.option("--to", "to_protocol", required=True, type=click.Choice(["a2a", "mcp", "autogen", "langgraph", "crewai"]))
@click.option("--output", "output_file", help="Output file")
@click.pass_context
def protocol_convert(ctx, input_file, from_protocol, to_protocol, output_file):
    """Convert message between protocols"""
    async def _convert():
        try:
            with open(input_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[red]Failed to read input file: {e}[/red]")
            return
        
        # Convert
        from src.protocols.unified.protocol import UnifiedProtocol
        
        protocol = UnifiedProtocol()
        
        # Get adapter
        from_adapter = protocol.get_adapter(ProtocolType(from_protocol))
        to_adapter = protocol.get_adapter(ProtocolType(to_protocol))
        
        if not from_adapter or not to_adapter:
            console.print("[red]Protocol adapter not found[/red]")
            return
        
        # Convert
        unified = await from_adapter.to_unified(data)
        output = await to_adapter.from_unified(unified)
        
        # Output
        output_json = json.dumps(output, indent=2)
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(output_json)
            console.print(f"[green]Converted message saved to {output_file}[/green]")
        else:
            console.print(output_json)
    
    asyncio.run(_convert())


@protocol.command("info")
@click.pass_context
def protocol_info(ctx):
    """Show protocol information"""
    table = Table(title="Supported Protocols")
    table.add_column("Protocol", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("Features", style="green")
    
    protocols = [
        ("Google A2A", "1.0.0", "Agent cards, tasks/send, tasks/status, tasks/cancel"),
        ("MCP", "1.0.0", "Tool calling, resources, sampling"),
        ("AutoGen", "0.7.5", "Group chat, code execution, orchestration"),
        ("LangGraph", "0.2.0", "Workflows, state management, checkpoints"),
        ("CrewAI", "0.100.0", "Crews, collaboration, tool sharing"),
    ]
    
    for name, version, features in protocols:
        table.add_row(name, version, features)
    
    console.print(table)
    
    console.print("\n[cyan]Unified Protocol Features:[/cyan]")
    console.print("- Multi-protocol support with automatic detection")
    console.print("- Unified message format across all protocols")
    console.print("- Protocol adapters for seamless conversion")
    console.print("- Task management with retry logic")
    console.print("- Workflow orchestration with state management")
    console.print("- Human-in-the-loop integration")
    console.print("- Security with DID and signatures")


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()


