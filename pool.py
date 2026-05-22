"""Warm-pool for Droid CLI process startup.

Background
----------
``droid exec`` is per-turn (no daemon mode). Each Matrix message spawns a
fresh ``droid exec`` invocation. On a cold box, that takes ~10 s before the
first model token. Measurements on Contabo vmi3318318 (2026-05-22):

    Turn 1 (cold):  9.0 s
    Turn 2:         8.0 s   (OS page-cache + DNS warm)
    Turn 3:         7.0 s   (more pages warm)

The OS warms itself naturally, but we can drive it harder by **pre-spawning
``droid exec --list-tools`` calls in the background**. ``--list-tools`` is a
cheap probe (no LLM call) that:

  * Boots the Node.js binary into the page cache.
  * Authenticates against Factory API (cached).
  * Loads the tool catalog.
  * Resolves DNS for the custom model base URL.
  * Establishes a TLS session ticket to ``llm.garza.online``.
  * Reads ``~/.factory/settings.json`` and validates customModels.

After a few warmups the next *real* ``droid exec`` benefits from all of that
cached state without the pool holding a long-lived subprocess. This is the
correct shape for Droid's per-turn model — distinct from a JSON-RPC daemon
pool.

The pool exposes a trivial interface:

    pool = WarmPool(droid_config, min_warm=2, refresh_interval_s=240)
    await pool.start()
    # ... time passes, real turns happen via DroidHarness.add_user_message_stream ...
    await pool.stop()

It runs entirely in the background and has no direct call from the manager
loop. Its only job is to keep the OS + Factory client state warm. The
``DroidHarness`` per-turn spawns automatically benefit.

Configuration knobs (read from ``DroidConfig`` plus env overrides):

  * ``pool_min_warm``       — number of concurrent warmup processes
  * ``pool_refresh_seconds``— how often each slot re-runs ``--list-tools``
  * ``pool_initial_burst``  — extra warmups on first start to prime aggressively
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from .config import DroidConfig

logger = logging.getLogger("droid_manager.pool")

DEFAULT_MIN_WARM = 2
DEFAULT_REFRESH_S = 240          # re-warm each slot every 4 min
DEFAULT_INITIAL_BURST = 3        # initial parallel warmups
DEFAULT_TIMEOUT_S = 25           # max time a single --list-tools call may take


class WarmPool:
    """Pre-spawn ``droid exec --list-tools`` to keep OS + Factory caches warm."""

    def __init__(
        self,
        config: DroidConfig,
        *,
        min_warm: int | None = None,
        refresh_interval_s: int | None = None,
        initial_burst: int | None = None,
    ) -> None:
        self._cfg = config
        self._min_warm = int(min_warm if min_warm is not None
                             else os.environ.get("HICLAW_DROID_POOL_MIN_WARM",
                                                 DEFAULT_MIN_WARM))
        self._refresh_s = int(refresh_interval_s if refresh_interval_s is not None
                              else os.environ.get("HICLAW_DROID_POOL_REFRESH_S",
                                                  DEFAULT_REFRESH_S))
        self._initial_burst = int(initial_burst if initial_burst is not None
                                  else os.environ.get("HICLAW_DROID_POOL_BURST",
                                                      DEFAULT_INITIAL_BURST))
        self._stopped = asyncio.Event()
        self._slot_tasks: list[asyncio.Task] = []
        self._stats = {
            "warmups_started": 0,
            "warmups_succeeded": 0,
            "warmups_failed": 0,
            "avg_warmup_ms": 0.0,
            "last_warmup_at": 0.0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    async def start(self) -> None:
        """Kick off the initial burst + start the long-lived per-slot loops."""
        if self._slot_tasks:
            return
        logger.info(
            "WarmPool starting: min_warm=%d refresh_s=%d initial_burst=%d "
            "binary=%s model=%s",
            self._min_warm,
            self._refresh_s,
            self._initial_burst,
            self._cfg.binary,
            self._cfg.model,
        )

        # Initial parallel burst — fire N warmups simultaneously to prime fast
        burst_tasks = [
            asyncio.create_task(self._warmup_once(label=f"burst-{i}"))
            for i in range(self._initial_burst)
        ]
        # Don't await burst; let it run while the slot loops also start

        # Per-slot long-lived loops: each refreshes every refresh_interval_s
        for slot in range(self._min_warm):
            t = asyncio.create_task(
                self._slot_loop(slot_id=slot),
                name=f"WarmPool.slot-{slot}",
            )
            self._slot_tasks.append(t)

        # Optional: await burst so caller knows the pool is hot
        try:
            await asyncio.wait_for(asyncio.gather(*burst_tasks, return_exceptions=True),
                                   timeout=DEFAULT_TIMEOUT_S * 2)
            logger.info(
                "WarmPool initial burst done — succeeded=%d failed=%d avg_ms=%.0f",
                self._stats["warmups_succeeded"],
                self._stats["warmups_failed"],
                self._stats["avg_warmup_ms"],
            )
        except asyncio.TimeoutError:
            logger.warning("WarmPool initial burst timed out; slot loops continue")

    async def stop(self) -> None:
        """Cancel all warmup loops and wait for them to drain."""
        self._stopped.set()
        for t in self._slot_tasks:
            t.cancel()
        await asyncio.gather(*self._slot_tasks, return_exceptions=True)
        self._slot_tasks.clear()
        logger.info("WarmPool stopped — final stats: %s", self._stats)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _slot_loop(self, slot_id: int) -> None:
        """Long-lived task: re-warm slot `slot_id` every refresh_interval_s."""
        # Stagger initial wait so slots fire out of phase
        await asyncio.sleep(self._refresh_s * slot_id / max(1, self._min_warm))
        while not self._stopped.is_set():
            await self._warmup_once(label=f"slot-{slot_id}")
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._refresh_s)
                # If we got here, stop was called
                return
            except asyncio.TimeoutError:
                continue

    async def _warmup_once(self, *, label: str) -> bool:
        """Run one ``droid exec --list-tools`` to warm OS + Factory caches.

        Returns True on success, False on failure or timeout.
        """
        self._stats["warmups_started"] += 1
        t0 = time.monotonic()
        argv = [
            self._cfg.binary,
            "exec",
            "--list-tools",
            "-m",
            self._cfg.model,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=os.environ.copy(),
            )
            rc = await asyncio.wait_for(proc.wait(), timeout=DEFAULT_TIMEOUT_S)
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._stats["last_warmup_at"] = time.time()
            if rc == 0:
                self._stats["warmups_succeeded"] += 1
                n = self._stats["warmups_succeeded"]
                # Online avg: avg_n = avg_{n-1} + (x - avg_{n-1})/n
                self._stats["avg_warmup_ms"] = (
                    self._stats["avg_warmup_ms"]
                    + (elapsed_ms - self._stats["avg_warmup_ms"]) / n
                )
                logger.info(
                    "WarmPool %s ok in %.0fms (n=%d avg=%.0fms)",
                    label, elapsed_ms, n, self._stats["avg_warmup_ms"],
                )
                return True
            self._stats["warmups_failed"] += 1
            logger.warning("WarmPool %s exit=%d in %.0fms", label, rc, elapsed_ms)
            return False
        except asyncio.TimeoutError:
            self._stats["warmups_failed"] += 1
            logger.warning("WarmPool %s timed out after %ds", label, DEFAULT_TIMEOUT_S)
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass
            return False
        except Exception as exc:
            self._stats["warmups_failed"] += 1
            logger.exception("WarmPool %s crashed: %s", label, exc)
            return False
