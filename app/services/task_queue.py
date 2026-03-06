"""Simple in-memory async task queue.

Allows submitting generation tasks for background processing, polling status,
and retrieving results.  Uses asyncio for concurrency -- no external broker
required.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, Optional

from app.models.schemas import TaskState, TaskStatus

logger = logging.getLogger(__name__)


class TaskQueue:
    """Lightweight in-memory async task queue."""

    def __init__(self, max_concurrent: int = 10) -> None:
        self._tasks: Dict[str, TaskStatus] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        coro_factory: Callable[[], Coroutine],
        task_id: str | None = None,
    ) -> TaskStatus:
        """Schedule a coroutine for background execution.

        Parameters
        ----------
        coro_factory : A zero-arg callable that returns a coroutine.
        task_id      : Optional custom ID; generated if omitted.

        Returns
        -------
        TaskStatus with the task_id you can use to poll.
        """
        tid = task_id or str(uuid.uuid4())
        status = TaskStatus(
            task_id=tid,
            state=TaskState.pending,
            created_at=datetime.now(timezone.utc),
        )
        self._tasks[tid] = status

        # Fire-and-forget in the running event loop
        asyncio.ensure_future(self._run(tid, coro_factory))
        return status

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[Any]:
        return self._results.get(task_id)

    def all_tasks(self) -> list[TaskStatus]:
        return list(self._tasks.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(self, task_id: str, coro_factory: Callable[[], Coroutine]) -> None:
        status = self._tasks[task_id]
        async with self._semaphore:
            status.state = TaskState.running
            try:
                result = await coro_factory()
                self._results[task_id] = result
                status.state = TaskState.completed
                # If result is a Pydantic model, store its dict form too
                if hasattr(result, "model_dump"):
                    status.result = result.model_dump()
                elif hasattr(result, "dict"):
                    status.result = result.dict()
                else:
                    status.result = result
            except Exception as exc:
                logger.error("Task %s failed: %s", task_id, exc, exc_info=True)
                status.state = TaskState.failed
                status.error = str(exc)
            finally:
                status.completed_at = datetime.now(timezone.utc)
