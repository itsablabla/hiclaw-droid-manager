"""Per-turn subprocess wrapper around ``droid exec`` (rewritten for real protocol).

The original harness.py assumed Droid CLI 0.131.0 spoke a long-lived JSON-RPC
daemon protocol with methods like ``droid.initialize_session`` and
``droid.add_user_message``. Empirical testing on Contabo vmi3318318 showed
Droid CLI rejects every such request with ``Invalid JSON-RPC message``.

What Droid actually does (verified):

    $ echo "Reply HELLO" | droid exec --output-format stream-json \
        -m custom:Garza-Haiku-4.5-0
    {"type":"system","subtype":"init","session_id":"<uuid>","model":"...","tools":[...]}
    {"type":"message","role":"user","text":"Reply HELLO","session_id":"<uuid>"}
    {"type":"message","role":"assistant","text":"HELLO","session_id":"<uuid>"}
    {"type":"completion","finalText":"HELLO","numTurns":1,"durationMs":2494,
     "session_id":"<uuid>","usage":{"input_tokens":14848,"output_tokens":53,...}}

So each turn is a fresh ``droid exec`` invocation. Multi-turn continuity is
achieved by passing ``--session-id <prev_session_id>`` to resume. This is the
"Camp A" pattern used by Vibe Kanban, agents-js, and Gobby.

The new harness keeps the same public API as the original (so manager.py
needs no changes) but replaces the JSON-RPC plumbing with a per-turn spawner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from .config import DroidConfig

logger = logging.getLogger("droid_manager.harness")


@dataclass
class HarnessEvent:
    """Normalised event surfaced to the manager loop.

    ``kind`` is one of:
      * ``"text_delta"``         — partial assistant text (``content``)
      * ``"text_complete"``      — full assistant message (``content``)
      * ``"tool_use"``           — agent is calling a tool (``tool``, ``args``)
      * ``"tool_result"``        — tool returned (``tool``, ``result``)
      * ``"turn_complete"``      — turn ended; safe to reply to user
      * ``"permission_request"`` — droid wants approval
      * ``"ask_user"``           — droid wants out-of-band input
      * ``"error"``              — error notification (``message``)
      * ``"token_usage"``        — token usage update (``input``, ``output``)
    """

    kind: str
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return str(self.payload.get("content") or "")


@dataclass
class HarnessRequest:
    """Kept for API compatibility with old code; unused in the per-turn model."""

    id: str
    method: str
    params: dict[str, Any]


PermissionResolver = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class DroidHarness:
    """Per-turn ``droid exec`` spawner with session resume.

    Public API (kept stable from the original JSON-RPC version):

        async with DroidHarness(cfg) as harness:
            session_id = await harness.initialize_session(name="matrix:!room")
            async for ev in harness.add_user_message_stream(text="hi",
                                                            session_id=session_id):
                ...

    Internally:
      * ``initialize_session`` is a no-op that returns a placeholder until the
        first real turn runs (Droid hands us a real session_id with the
        ``system/init`` event of the first ``droid exec`` call).
      * ``add_user_message_stream`` spawns ONE ``droid exec`` per call.
      * Subsequent calls pass ``--session-id <prev>`` to resume.
    """

    def __init__(
        self,
        config: DroidConfig,
        *,
        permission_resolver: PermissionResolver | None = None,
        ask_user_resolver: PermissionResolver | None = None,
    ) -> None:
        self._cfg = config
        # Session id last assigned by *this* harness — used as a default for
        # add_user_message_stream when the caller doesn't pass session_id.
        self._session_id: Optional[str] = None
        # When initialize_session is called with resume_session_id we remember
        # it so the *first* turn uses --session-id <id>.
        self._pending_resume: Optional[str] = None
        self._permission_resolver = permission_resolver
        self._ask_user_resolver = ask_user_resolver
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle (no-ops in per-turn model)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DroidHarness":
        await self.start()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        """Verify droid binary is on PATH. Does not spawn a long-lived process."""
        if not shutil.which(self._cfg.binary):
            raise FileNotFoundError(
                f"droid binary '{self._cfg.binary}' not found on PATH. "
                f"Install with: curl -fsSL https://app.factory.ai/cli | sh"
            )
        if not os.environ.get(self._cfg.api_key_env):
            logger.warning(
                "%s not set — droid exec may fail to authenticate. Generate "
                "an API key at app.factory.ai/settings/api-keys.",
                self._cfg.api_key_env,
            )
        try:
            self._cfg.cwd.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("cwd mkdir failed (%s); droid may still run", exc)
        logger.info(
            "DroidHarness ready (per-turn spawn mode, binary=%s, model=%s, cwd=%s)",
            shutil.which(self._cfg.binary),
            self._cfg.model,
            self._cfg.cwd,
        )

    async def stop(self) -> None:
        self._stopped = True

    # ------------------------------------------------------------------
    # Public RPC API (preserves the original interface)
    # ------------------------------------------------------------------

    async def initialize_session(
        self,
        *,
        name: str | None = None,
        resume_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Reserve a session for the next turn.

        Droid CLI doesn't have a separate "create session" call — sessions are
        born when ``droid exec`` runs without ``--session-id``. So this method
        only stashes an optional resume id; the real session_id is filled in
        by the first ``add_user_message_stream`` call.

        Returns a placeholder string. The real id flows through the
        ``HarnessEvent.session_id`` field on the first turn.
        """
        if resume_session_id:
            self._pending_resume = resume_session_id
            self._session_id = resume_session_id
            logger.info("DroidHarness will resume session %s on next turn", resume_session_id)
            return resume_session_id
        # Use the placeholder; the real id is assigned on first turn
        self._pending_resume = None
        self._session_id = "pending"
        logger.info("DroidHarness will create a fresh session on next turn")
        return "pending"

    async def fork_session(self, source_session_id: str) -> str:
        """Fork an existing session using ``droid exec --fork <id>``.

        Returns the new session id. Implementation: run a tiny prompt with
        ``--fork`` and capture the system/init event.
        """
        sid = await self._run_one_turn(
            text="(continue)",
            session_id=None,
            extra_argv=["--fork", source_session_id],
            consume_all=False,
        )
        return sid

    async def interrupt(self, session_id: str | None = None) -> None:
        """No-op: Droid CLI doesn't support out-of-band interrupt; the
        subprocess can be cancelled by closing its stdout reader."""
        logger.debug("interrupt() called for session %s (no-op in per-turn mode)",
                     session_id or self._session_id)

    async def compact_history(self, prompt: str | None = None) -> dict[str, Any]:
        """Compact via a meta-prompt — there's no dedicated CLI for this."""
        return {"status": "noop", "reason": "compact not implemented in per-turn mode"}

    async def list_tools(self) -> list[dict[str, Any]]:
        """Run ``droid exec --list-tools`` and parse the output."""
        argv = [self._cfg.binary, "exec", "--list-tools", "-m", self._cfg.model]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            return [{"name": line.strip()} for line in out.decode().splitlines()
                    if line.strip() and not line.startswith(("Available", "Autonomy"))]
        except Exception as exc:
            logger.warning("list_tools failed: %s", exc)
            return []

    async def add_user_message_stream(
        self,
        *,
        text: str,
        attachments: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[HarnessEvent]:
        """Spawn ONE ``droid exec`` for this turn and stream events.

        If ``session_id`` is provided AND is not the placeholder "pending",
        passes ``--session-id`` to resume. Otherwise starts a fresh session.
        """
        resume = None
        if session_id and session_id != "pending":
            resume = session_id
        elif self._pending_resume:
            resume = self._pending_resume
            self._pending_resume = None  # only use once

        argv = [
            self._cfg.binary,
            "exec",
            "--output-format",
            "stream-json",
            "--auto",
            self._cfg.auto_level,
            "--cwd",
            str(self._cfg.cwd),
            "-m",
            self._cfg.model,
        ]
        if self._cfg.reasoning_effort:
            argv += ["-r", self._cfg.reasoning_effort]
        if resume:
            argv += ["--session-id", resume]
        for tag in (self._cfg.tags or []):
            argv += ["--tag", tag]
        if self._cfg.log_group_id:
            argv += ["--log-group-id", self._cfg.log_group_id]
        if self._cfg.append_system_prompt_file:
            argv += ["--append-system-prompt-file", str(self._cfg.append_system_prompt_file)]
        if self._cfg.skip_permissions_unsafe:
            argv.append("--skip-permissions-unsafe")
        # Prompt is positional, LAST
        argv.append(text)

        logger.info("droid exec: %s ... '%s'",
                    " ".join(argv[:-1]),
                    text[:60].replace("\n", " "))

        env = os.environ.copy()

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cfg.cwd),
                env=env,
            )
        except Exception as exc:
            logger.exception("failed to spawn droid: %s", exc)
            yield HarnessEvent("error", session_id or "", {"message": f"spawn failed: {exc}"})
            return

        # Stream lines from stdout, parse JSON, yield HarnessEvent.
        # Drain stderr in parallel so the process doesn't deadlock on a full
        # stderr pipe.
        stderr_lines: list[str] = []

        async def drain_stderr() -> None:
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                s = line.decode("utf-8", errors="replace").rstrip()
                stderr_lines.append(s)
                logger.debug("droid stderr: %s", s)

        stderr_task = asyncio.create_task(drain_stderr())

        new_session_id = resume or ""
        saw_assistant_message = False
        accumulated_text = ""

        assert proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace").strip() or "{}")
                except json.JSONDecodeError:
                    continue

                kind = msg.get("type")
                sid_in_msg = msg.get("session_id")
                if sid_in_msg:
                    new_session_id = sid_in_msg
                    if not self._session_id or self._session_id == "pending":
                        self._session_id = sid_in_msg

                if kind == "system" and msg.get("subtype") == "init":
                    # init event - just record the session id and continue
                    continue

                if kind == "message":
                    role = msg.get("role")
                    text_body = msg.get("text") or ""
                    if role == "assistant" and text_body:
                        # Distinguish first chunk vs subsequent
                        if not saw_assistant_message:
                            saw_assistant_message = True
                        accumulated_text = text_body  # Droid sends full text per message
                        yield HarnessEvent(
                            "text_complete",
                            new_session_id,
                            {"content": text_body},
                        )
                    # role=user is the echo of our input; skip it
                    continue

                if kind == "tool_use" or (kind == "message" and msg.get("role") == "tool"):
                    yield HarnessEvent(
                        "tool_use",
                        new_session_id,
                        {
                            "tool": msg.get("name") or msg.get("tool", ""),
                            "args": msg.get("input") or msg.get("arguments", {}),
                        },
                    )
                    continue

                if kind == "tool_result":
                    yield HarnessEvent(
                        "tool_result",
                        new_session_id,
                        {
                            "tool": msg.get("name") or msg.get("tool", ""),
                            "result": msg.get("output") or msg.get("result", ""),
                        },
                    )
                    continue

                if kind == "completion":
                    usage = msg.get("usage") or {}
                    if usage:
                        yield HarnessEvent(
                            "token_usage",
                            new_session_id,
                            {
                                "input": usage.get("input_tokens", 0),
                                "output": usage.get("output_tokens", 0),
                            },
                        )
                    yield HarnessEvent("turn_complete", new_session_id, dict(msg))
                    break

                if kind == "error":
                    yield HarnessEvent(
                        "error",
                        new_session_id,
                        {"message": msg.get("message", str(msg))},
                    )
                    continue

                logger.debug("droid: unhandled message: %s", msg)

            # Wait for process to exit so we can collect rc
            rc = await proc.wait()
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

            if rc != 0:
                msg = "\n".join(stderr_lines[-5:]) or f"exit code {rc}"
                logger.warning("droid exec exited non-zero: %s", msg)
                if not saw_assistant_message:
                    yield HarnessEvent(
                        "error",
                        new_session_id,
                        {"message": f"droid exec failed: {msg}"},
                    )
                    yield HarnessEvent("turn_complete", new_session_id, {})
        except Exception as exc:
            logger.exception("droid stream pump error: %s", exc)
            stderr_task.cancel()
            try:
                proc.terminate()
            except Exception:
                pass
            yield HarnessEvent("error", new_session_id, {"message": str(exc)})
            yield HarnessEvent("turn_complete", new_session_id, {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_one_turn(
        self,
        *,
        text: str,
        session_id: str | None,
        extra_argv: list[str] | None = None,
        consume_all: bool = True,
    ) -> str:
        """Helper for fork_session — run one short droid call, return the new session_id."""
        argv = [
            self._cfg.binary, "exec", "--output-format", "stream-json",
            "--auto", "low", "--cwd", str(self._cfg.cwd),
            "-m", self._cfg.model,
        ]
        if session_id:
            argv += ["--session-id", session_id]
        if extra_argv:
            argv += extra_argv
        argv.append(text)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        new_sid = ""
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode().strip() or "{}")
            except json.JSONDecodeError:
                continue
            if msg.get("session_id") and not new_sid:
                new_sid = msg["session_id"]
            if msg.get("type") == "completion":
                if not consume_all:
                    break
        await proc.wait()
        return new_sid
