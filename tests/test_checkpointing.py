from __future__ import annotations

from pathlib import Path
from typing import Any
import asyncio
import multiprocessing

import pytest
from langgraph.graph import END, StateGraph

from core.checkpointing import (
    CheckpointSettings,
    CheckpointConfigurationError,
    RunAlreadyActiveError,
    RunAlreadyExistsError,
    FileRunLeaseManager,
    open_checkpointer,
)
from core.audit import open_audit_store
from swarm.medical_swarm_graph import MedicalSwarmGraph
from swarm.medical_swarm_state import MedicalSwarmState
from swarm.shared_context import Contribution, SharedContext, SubTask


def _hold_file_lease(directory: str, ready, release) -> None:
    async def hold() -> None:
        manager = FileRunLeaseManager(Path(directory))
        async with manager.claim("cross-process-run"):
            ready.set()
            await asyncio.to_thread(release.wait)

    asyncio.run(hold())


class RecoveryProbeGraph(MedicalSwarmGraph):
    """Small graph exercising MedicalSwarmGraph's public run interface."""

    def __init__(
        self,
        checkpointer: Any,
        *,
        fail_once: bool = False,
        block_prepare: bool = False,
    ):
        self.checkpointer = checkpointer
        self.fail_once = fail_once
        self.failure_count = 0
        self.block_prepare = block_prepare
        self.prepare_started = asyncio.Event()
        self.prepare_release = asyncio.Event()
        self.enable_swarm = False
        self.enable_short_term_memory = False
        self.enable_long_term_memory = False
        self.swarm_timeout = 10.0
        self.worker_pool = []
        self._compiled_graph = self.build_graph()

    def build_graph(self):
        graph = StateGraph(MedicalSwarmState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("finish", self._finish)
        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=self.checkpointer)

    async def _prepare(self, state: MedicalSwarmState):
        if self.block_prepare:
            self.prepare_started.set()
            await self.prepare_release.wait()
        return {"assessment": {"prepared": True}}

    async def _finish(self, state: MedicalSwarmState):
        if self.fail_once and self.failure_count == 0:
            self.failure_count += 1
            raise RuntimeError("simulated process interruption")
        return {
            "result": {
                "answer": "recovered",
                "prepared": state["assessment"]["prepared"],
            }
        }


@pytest.mark.asyncio
async def test_runs_use_run_id_as_checkpoint_thread_not_session_id(tmp_path: Path):
    settings = CheckpointSettings(
        backend="sqlite",
        sqlite_path=tmp_path / "runs.sqlite3",
        allow_plaintext=True,
    )

    async with open_checkpointer(settings) as checkpointer:
        graph = RecoveryProbeGraph(checkpointer)
        first = await graph.ainvoke(
            {"question": "first", "context": {}, "session_id": "session-a"}
        )
        second = await graph.ainvoke(
            {"question": "second", "context": {}, "session_id": "session-a"}
        )

        assert first["session_id"] == second["session_id"] == "session-a"
        assert first["run_id"] != second["run_id"]
        assert await graph.get_checkpoint(first["run_id"]) is not None
        assert await graph.get_checkpoint(second["run_id"]) is not None


@pytest.mark.asyncio
async def test_sqlite_checkpoint_resumes_after_graph_is_recreated(tmp_path: Path):
    database_path = tmp_path / "recovery.sqlite3"
    settings = CheckpointSettings(
        backend="sqlite",
        sqlite_path=database_path,
        allow_plaintext=True,
    )

    async with open_checkpointer(settings) as checkpointer:
        graph = RecoveryProbeGraph(checkpointer, fail_once=True)
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            await graph.ainvoke(
                {
                    "run_id": "run-to-resume",
                    "question": "recover me",
                    "context": {},
                    "session_id": "session-a",
                }
            )

    async with open_checkpointer(settings) as checkpointer:
        restarted_graph = RecoveryProbeGraph(checkpointer)
        final_state = await restarted_graph.resume("run-to-resume")

        assert final_state["result"] == {
            "answer": "recovered",
            "prepared": True,
        }
        history = await restarted_graph.list_checkpoints("run-to-resume")
        assert len(history) >= 3
        assert history[0].status == "completed"


@pytest.mark.asyncio
async def test_same_run_cannot_execute_concurrently():
    settings = CheckpointSettings(backend="memory")
    async with open_checkpointer(settings) as checkpointer:
        graph = RecoveryProbeGraph(checkpointer, block_prepare=True)
        first = asyncio.create_task(
            graph.ainvoke(
                {
                    "run_id": "exclusive-run",
                    "question": "first",
                    "context": {},
                    "session_id": "session-a",
                }
            )
        )
        await graph.prepare_started.wait()

        with pytest.raises(RunAlreadyActiveError, match="exclusive-run"):
            await graph.ainvoke(
                {
                    "run_id": "exclusive-run",
                    "question": "second",
                    "context": {},
                    "session_id": "session-a",
                }
            )

        graph.prepare_release.set()
        await first


@pytest.mark.asyncio
async def test_existing_run_id_must_be_resumed_not_restarted():
    settings = CheckpointSettings(backend="memory")
    async with open_checkpointer(settings) as checkpointer:
        graph = RecoveryProbeGraph(checkpointer)
        initial_state = {
            "run_id": "existing-run",
            "question": "first",
            "context": {},
            "session_id": "session-a",
        }
        await graph.ainvoke(initial_state)

        with pytest.raises(RunAlreadyExistsError, match="existing-run"):
            await graph.ainvoke(initial_state)


@pytest.mark.asyncio
async def test_completed_run_can_replay_from_selected_checkpoint():
    settings = CheckpointSettings(backend="memory")
    async with open_checkpointer(settings) as checkpointer:
        graph = RecoveryProbeGraph(checkpointer)
        await graph.ainvoke(
            {
                "run_id": "time-travel-run",
                "question": "replay me",
                "context": {},
                "session_id": "session-a",
            }
        )
        history = await graph.list_checkpoints("time-travel-run")
        before_finish = next(
            checkpoint
            for checkpoint in history
            if checkpoint.next_nodes == ("finish",)
        )
        original_checkpoint_ids = {
            checkpoint.checkpoint_id for checkpoint in history
        }

        replayed = await graph.resume(
            "time-travel-run",
            before_finish.checkpoint_id,
        )

        assert replayed["result"]["answer"] == "recovered"
        replay_history = await graph.list_checkpoints("time-travel-run")
        replay_branch = [
            checkpoint
            for checkpoint in replay_history
            if checkpoint.checkpoint_id not in original_checkpoint_ids
        ]
        assert any(
            checkpoint.parent_checkpoint_id == before_finish.checkpoint_id
            for checkpoint in replay_branch
        )


@pytest.mark.asyncio
async def test_sqlite_checkpoint_can_encrypt_medical_state_at_rest(tmp_path: Path):
    database_path = tmp_path / "encrypted.sqlite3"
    secret_question = "unique-sensitive-symptom-9f1e"
    settings = CheckpointSettings(
        backend="sqlite",
        sqlite_path=database_path,
        encryption_key="0123456789abcdef0123456789abcdef",
    )

    async with open_checkpointer(settings) as checkpointer:
        graph = RecoveryProbeGraph(checkpointer)
        await graph.ainvoke(
            {
                "run_id": "encrypted-run",
                "question": secret_question,
                "context": {"medical_history": "sensitive-history-2b7c"},
                "session_id": "session-a",
            }
        )

    database_bytes = database_path.read_bytes()
    assert secret_question.encode("utf-8") not in database_bytes
    assert b"sensitive-history-2b7c" not in database_bytes

    async with open_checkpointer(settings) as checkpointer:
        restarted_graph = RecoveryProbeGraph(checkpointer)
        checkpoint = await restarted_graph.get_checkpoint("encrypted-run")
        assert checkpoint is not None
        assert checkpoint.values["question"] == secret_question


def test_durable_checkpoint_backend_fails_closed_without_encryption(tmp_path: Path):
    with pytest.raises(CheckpointConfigurationError, match="CHECKPOINT_AES_KEY"):
        CheckpointSettings(
            backend="sqlite",
            sqlite_path=tmp_path / "plaintext.sqlite3",
        )


@pytest.mark.asyncio
async def test_audit_attempts_survive_restart_and_are_encrypted(tmp_path: Path):
    settings = CheckpointSettings(
        backend="sqlite",
        sqlite_path=tmp_path / "checkpoints.sqlite3",
        encryption_key="0123456789abcdef0123456789abcdef",
    )
    payload = {
        "run": {"run_id": "audit-run", "started_at": "2026-08-02T00:00:00"},
        "events": [{"name": "clinical-step", "output": "sensitive-audit-data"}],
    }

    async with open_audit_store(settings) as audit_store:
        await audit_store.save_attempt("audit-run", "attempt-a", payload)

    audit_path = tmp_path / "checkpoints-audit.sqlite3"
    assert b"sensitive-audit-data" not in audit_path.read_bytes()

    async with open_audit_store(settings) as audit_store:
        attempts = await audit_store.get_attempts("audit-run")

    assert attempts == [{"attempt_id": "attempt-a", **payload}]


@pytest.mark.asyncio
async def test_audit_effect_claim_survives_restart(tmp_path: Path):
    settings = CheckpointSettings(
        backend="sqlite",
        sqlite_path=tmp_path / "checkpoints.sqlite3",
        encryption_key="0123456789abcdef0123456789abcdef",
    )

    async with open_audit_store(settings) as audit_store:
        assert await audit_store.claim_effect("run-a", "long_term_memory") is True
        await audit_store.complete_effect("run-a", "long_term_memory", "completed")

    async with open_audit_store(settings) as audit_store:
        assert await audit_store.claim_effect("run-a", "long_term_memory") is False


def test_sqlite_file_lease_excludes_another_process(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_file_lease,
        args=(str(tmp_path), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=10)
        manager = FileRunLeaseManager(tmp_path)

        async def contend() -> None:
            with pytest.raises(RunAlreadyActiveError, match="cross-process-run"):
                async with manager.claim("cross-process-run"):
                    pass

        asyncio.run(contend())
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_shared_context_checkpoint_round_trip_is_primitive():
    context = SharedContext(session_id="session-a")
    task = SubTask(
        id="risk",
        type="risk_assessment",
        description="Assess urgency",
        assigned_agent="consultation_agent",
    )
    context.add_subtask(task)
    context.start_subtask(task.id)
    context.complete_subtask(
        task.id,
        "consultation_agent",
        {"risk_level": "low"},
        confidence=0.8,
    )
    context.agent_contributions["consultation_agent"].append(
        Contribution(
            agent_id="consultation_agent",
            subtask_id="risk",
            result={"note": "reviewed"},
        )
    )

    snapshot = context.to_checkpoint()
    restored = SharedContext.from_checkpoint(snapshot)

    assert isinstance(snapshot, dict)
    assert snapshot["schema_version"] == 1
    assert restored.session_id == "session-a"
    assert restored.get_subtask("risk").result == {"risk_level": "low"}
    assert len(restored.get_contributions("consultation_agent")) == 2
