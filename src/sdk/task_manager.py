"""
Task Manager

Manages task lifecycle and execution across different protocols.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from ..protocols.unified.message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from ..protocols.unified.agent_card import AgentCard
from ..protocols.unified.task import Task, TaskResult, TaskPriority


class TaskManager:
    """
    Task Manager
    
    Manages task lifecycle across different protocols.
    
    Features:
    - Task creation and execution
    - Status tracking
    - Retry logic
    - Callbacks
    - Batch operations
    
    Example:
        >>> manager = TaskManager()
        >>> 
        >>> # Create task
        >>> task = await manager.create_task(
        ...     agent_id="agent-123",
        ...     task_data={"prompt": "Process data"},
        ... )
        >>> 
        >>> # Wait for completion
        >>> result = await manager.wait_for_task(task.id)
        >>> 
        >>> # Batch execution
        >>> tasks = await manager.create_batch([
        ...     {"agent_id": "agent-1", "task_data": {"x": 1}},
        ...     {"agent_id": "agent-2", "task_data": {"x": 2}},
        ... ])
    """
    
    def __init__(
        self,
        default_timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Task Manager
        
        Args:
            default_timeout: Default task timeout
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
        """
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Task storage
        self._tasks: Dict[str, Task] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Logger
        self._logger = logging.getLogger(__name__)
    
    async def create_task(
        self,
        agent_id: str,
        task_data: Dict[str, Any],
        priority: int = 2,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        callback: Optional[Callable[[Task], None]] = None,
    ) -> Task:
        """
        Create and queue task
        
        Args:
            agent_id: Target agent ID
            task_data: Task data
            priority: Task priority (1-5)
            timeout: Task timeout
            max_retries: Max retry attempts
            callback: Completion callback
            
        Returns:
            Created task
        """
        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            name=task_data.get("name", "Unnamed Task"),
            description=task_data.get("description", ""),
            assignee_id=agent_id,
            input_data=task_data,
            priority=TaskPriority(priority),
            timeout=timeout or self.default_timeout,
            max_retries=max_retries or self.max_retries,
        )
        
        # Store task
        self._tasks[task.id] = task
        
        # Register callback
        if callback:
            if task.id not in self._callbacks:
                self._callbacks[task.id] = []
            self._callbacks[task.id].append(callback)
        
        self._logger.info(f"Created task: {task.id} for agent: {agent_id}")
        
        return task
    
    async def execute_task(self, task_id: str) -> TaskResult:
        """
        Execute task with retry logic
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskResult
        """
        task = self._tasks.get(task_id)
        if not task:
            return TaskResult(
                status=TaskStatus.FAILED,
                error=f"Task not found: {task_id}",
                error_code="NOT_FOUND",
            )
        
        # Execute with retries
        for attempt in range(task.max_retries + 1):
            try:
                task.retry_count = attempt
                task.update_status(TaskStatus.RUNNING)
                
                # Simulate execution (in production, call actual agent)
                await asyncio.sleep(0.1)
                
                # Create result
                result = TaskResult(
                    status=TaskStatus.COMPLETED,
                    output={
                        "message": "Task executed successfully",
                        "task_id": task.id,
                        "attempts": attempt + 1,
                    },
                    execution_time=0.1,
                )
                
                task.update_status(TaskStatus.COMPLETED, result)
                
                # Trigger callbacks
                await self._trigger_callbacks(task)
                
                return result
                
            except Exception as e:
                self._logger.error(f"Task execution attempt {attempt + 1} failed: {e}")
                
                if attempt < task.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    # All retries exhausted
                    result = TaskResult(
                        status=TaskStatus.FAILED,
                        error=str(e),
                        error_code="EXECUTION_ERROR",
                    )
                    task.update_status(TaskStatus.FAILED, result)
                    await self._trigger_callbacks(task)
                    return result
        
        # Should not reach here
        return TaskResult(
            status=TaskStatus.FAILED,
            error="Unexpected execution path",
        )
    
    async def _trigger_callbacks(self, task: Task) -> None:
        """Trigger task callbacks"""
        callbacks = self._callbacks.get(task.id, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                self._logger.error(f"Callback error: {e}")
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 1.0,
    ) -> Optional[Task]:
        """
        Wait for task completion
        
        Args:
            task_id: Task ID
            timeout: Maximum wait time
            poll_interval: Polling interval
            
        Returns:
            Completed task or None if timeout
        """
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        start_time = datetime.now()
        timeout = timeout or task.timeout
        
        while not task.is_complete():
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                self._logger.warning(f"Task {task_id} wait timeout")
                return None
            
            await asyncio.sleep(poll_interval)
        
        return task
    
    async def create_batch(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[Task]:
        """
        Create batch of tasks
        
        Args:
            tasks: List of task specifications
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of created tasks
        """
        created_tasks = []
        
        # Create all tasks
        for task_spec in tasks:
            task = await self.create_task(
                agent_id=task_spec["agent_id"],
                task_data=task_spec["task_data"],
                priority=task_spec.get("priority", 2),
            )
            created_tasks.append(task)
        
        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_limit(task: Task) -> TaskResult:
            async with semaphore:
                return await self.execute_task(task.id)
        
        # Execute all tasks
        await asyncio.gather(*[execute_with_limit(task) for task in created_tasks])
        
        return created_tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel task
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.is_complete():
            return False
        
        task.update_status(TaskStatus.CANCELLED)
        return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None
        """
        return self._tasks.get(task_id)
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        agent_id: Optional[str] = None,
    ) -> List[Task]:
        """
        List tasks
        
        Args:
            status: Filter by status
            agent_id: Filter by agent
            
        Returns:
            List of tasks
        """
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if agent_id:
            tasks = [t for t in tasks if t.assignee_id == agent_id]
        
        return tasks
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get task manager statistics
        
        Returns:
            Statistics dictionary
        """
        tasks = list(self._tasks.values())
        
        return {
            "total_tasks": len(tasks),
            "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
            "running": len([t for t in tasks if t.status == TaskStatus.RUNNING]),
            "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in tasks if t.status == TaskStatus.FAILED]),
        }


