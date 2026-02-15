"""
Workflow Engine

Orchestrates multi-agent workflows using LangGraph-style state management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..protocols.unified.message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from ..protocols.unified.task import Task, TaskResult
from .client import UnifiedClient
from .task_manager import TaskManager


@dataclass
class WorkflowStep:
    """Workflow step definition"""
    id: str
    name: str
    agent_id: str
    task_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None


@dataclass
class Workflow:
    """Workflow definition"""
    id: str
    name: str
    steps: Dict[str, WorkflowStep]
    initial_state: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_execution_order(self) -> List[str]:
        """Get steps in dependency order (topological sort)"""
        # Build dependency graph
        in_degree = {step_id: 0 for step_id in self.steps}
        dependents = {step_id: [] for step_id in self.steps}
        
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep in self.steps:
                    dependents[dep].append(step_id)
                    in_degree[step_id] += 1
        
        # Topological sort
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            step_id = queue.pop(0)
            order.append(step_id)
            
            for dependent in dependents[step_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return order


class WorkflowEngine:
    """
    Workflow Engine
    
    Orchestrates multi-agent workflows with state management.
    
    Features:
    - DAG-based workflow execution
    - State management
    - Conditional branching
    - Parallel execution
    - Checkpointing
    
    Example:
        >>> engine = WorkflowEngine()
        >>> 
        >>> # Define workflow
        >>> workflow = Workflow(
        ...     id="wf-1",
        ...     name="Data Processing",
        ...     steps={
        ...         "step1": WorkflowStep(
        ...             id="step1",
        ...             name="Fetch Data",
        ...             agent_id="agent-1",
        ...             task_data={"action": "fetch"},
        ...         ),
        ...         "step2": WorkflowStep(
        ...             id="step2",
        ...             name="Process Data",
        ...             agent_id="agent-2",
        ...             task_data={"action": "process"},
        ...             dependencies=["step1"],
        ...         ),
        ...     }
        ... )
        >>> 
        >>> # Execute
        >>> result = await engine.execute(workflow)
    """
    
    def __init__(
        self,
        task_manager: Optional[TaskManager] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Workflow Engine
        
        Args:
            task_manager: Task manager instance
            checkpoint_dir: Directory for checkpoints
        """
        self.task_manager = task_manager or TaskManager()
        self.checkpoint_dir = checkpoint_dir
        
        # Active workflows
        self._workflows: Dict[str, Workflow] = {}
        self._workflow_states: Dict[str, Dict[str, Any]] = {}
        self._workflow_results: Dict[str, Dict[str, TaskResult]] = {}
        
        self._logger = logging.getLogger(__name__)
    
    async def execute(
        self,
        workflow: Workflow,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute workflow
        
        Args:
            workflow: Workflow to execute
            initial_state: Initial state
            
        Returns:
            Final state
        """
        self._logger.info(f"Executing workflow: {workflow.name} ({workflow.id})")
        
        # Store workflow
        self._workflows[workflow.id] = workflow
        
        # Initialize state
        state = initial_state or workflow.initial_state.copy()
        self._workflow_states[workflow.id] = state
        self._workflow_results[workflow.id] = {}
        
        # Get execution order
        execution_order = workflow.get_execution_order()
        
        if not execution_order:
            self._logger.warning(f"No steps to execute in workflow: {workflow.id}")
            return state
        
        # Execute steps
        for step_id in execution_order:
            step = workflow.steps[step_id]
            
            # Check condition
            if step.condition and not step.condition(state):
                self._logger.info(f"Skipping step {step_id} - condition not met")
                continue
            
            # Execute step
            try:
                result = await self._execute_step(workflow.id, step, state)
                self._workflow_results[workflow.id][step_id] = result
                
                # Update state with result
                if result.is_success():
                    state[f"{step_id}_result"] = result.output
                    state["last_successful_step"] = step_id
                else:
                    state[f"{step_id}_error"] = result.error
                    state["last_failed_step"] = step_id
                    
                    # Handle failure
                    if step.on_failure:
                        # TODO: Implement failure handling
                        pass
                
            except Exception as e:
                self._logger.error(f"Step {step_id} execution failed: {e}")
                state[f"{step_id}_error"] = str(e)
        
        # Save final state
        self._workflow_states[workflow.id] = state
        
        self._logger.info(f"Workflow {workflow.id} completed")
        
        return state
    
    async def _execute_step(
        self,
        workflow_id: str,
        step: WorkflowStep,
        state: Dict[str, Any],
    ) -> TaskResult:
        """
        Execute workflow step
        
        Args:
            workflow_id: Workflow ID
            step: Step to execute
            state: Current state
            
        Returns:
            TaskResult
        """
        self._logger.info(f"Executing step: {step.name} ({step.id})")
        
        # Prepare task data with state context
        task_data = step.task_data.copy()
        task_data["workflow_id"] = workflow_id
        task_data["step_id"] = step.id
        task_data["state"] = state
        
        # Create and execute task
        task = await self.task_manager.create_task(
            agent_id=step.agent_id,
            task_data=task_data,
        )
        
        result = await self.task_manager.execute_task(task.id)
        
        return result
    
    async def execute_parallel(
        self,
        workflow: Workflow,
        initial_state: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute workflow with parallel step execution
        
        Args:
            workflow: Workflow to execute
            initial_state: Initial state
            max_concurrent: Maximum concurrent steps
            
        Returns:
            Final state
        """
        self._logger.info(f"Executing workflow in parallel: {workflow.name}")
        
        # Store workflow
        self._workflows[workflow.id] = workflow
        
        # Initialize state
        state = initial_state or workflow.initial_state.copy()
        self._workflow_states[workflow.id] = state
        self._workflow_results[workflow.id] = {}
        
        # Track completed steps
        completed: Set[str] = set()
        in_progress: Set[str] = set()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_step_async(step_id: str) -> None:
            """Execute step with semaphore"""
            async with semaphore:
                step = workflow.steps[step_id]
                
                # Check condition
                if step.condition and not step.condition(state):
                    completed.add(step_id)
                    return
                
                # Execute
                try:
                    result = await self._execute_step(workflow.id, step, state)
                    self._workflow_results[workflow.id][step_id] = result
                    
                    # Update state
                    if result.is_success():
                        state[f"{step_id}_result"] = result.output
                    else:
                        state[f"{step_id}_error"] = result.error
                        
                except Exception as e:
                    self._logger.error(f"Step {step_id} failed: {e}")
                    state[f"{step_id}_error"] = str(e)
                
                completed.add(step_id)
                in_progress.discard(step_id)
        
        # Execute steps respecting dependencies
        pending = set(workflow.steps.keys())
        
        while pending or in_progress:
            # Find ready steps
            ready = []
            for step_id in pending:
                step = workflow.steps[step_id]
                if all(dep in completed for dep in step.dependencies):
                    ready.append(step_id)
            
            # Start ready steps
            for step_id in ready:
                pending.discard(step_id)
                in_progress.add(step_id)
                asyncio.create_task(execute_step_async(step_id))
            
            # Wait for at least one step to complete
            if in_progress:
                await asyncio.sleep(0.1)
            elif pending:
                # Deadlock detected
                self._logger.error("Workflow deadlock detected")
                break
        
        return state
    
    def get_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow state
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            State or None
        """
        return self._workflow_states.get(workflow_id)
    
    def get_results(self, workflow_id: str) -> Optional[Dict[str, TaskResult]]:
        """
        Get workflow results
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Results or None
        """
        return self._workflow_results.get(workflow_id)


