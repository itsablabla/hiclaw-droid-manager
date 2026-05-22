#!/usr/bin/env python3
"""Idempotent patch applier for droid_manager/manager.py to add WarmPool.

This script is the safest way to inject the WarmPool wiring without
rewriting the whole manager.py (which is 308 lines and we don't want to
diverge from upstream). It edits manager.py in place.

Run inside the container:
    python3 /tmp/manager_patch.py /opt/hiclaw/droid-manager/src/droid_manager/manager.py
"""
import sys, re, pathlib

target = pathlib.Path(sys.argv[1])
src = target.read_text()

# 1. Add the WarmPool import after existing harness import
if "from .pool import WarmPool" not in src:
    src = src.replace(
        "from .harness import DroidHarness, HarnessEvent",
        "from .harness import DroidHarness, HarnessEvent\nfrom .pool import WarmPool",
        1,
    )

# 2. Instantiate WarmPool in DroidManager.__init__
# Find the line where self._harness is created and add self._pool after it
if "self._pool = WarmPool(" not in src:
    pattern = re.compile(
        r"(self\._harness = DroidHarness\([^)]*\)\n(?:\s+[^\n]*\n)*?)",
        re.DOTALL,
    )
    # Simpler: find self._harness = DroidHarness(...) call (multi-line) and inject after it
    idx = src.find("self._harness = DroidHarness(")
    if idx >= 0:
        # Walk forward to find the matching close-paren of the constructor
        depth = 0
        i = idx
        in_str = False
        while i < len(src):
            ch = src[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        # Move past the trailing \n
        end = i + 1
        if end < len(src) and src[end] == "\n":
            end += 1
        injection = "        self._pool = WarmPool(config.droid)\n"
        src = src[:end] + injection + src[end:]

# 3. Start the WarmPool inside run() right after self._harness.start()
if "await self._pool.start()" not in src:
    src = src.replace(
        "await self._harness.start()",
        "await self._harness.start()\n        await self._pool.start()",
        1,
    )

# 4. Stop the WarmPool in shutdown()
if "await self._pool.stop()" not in src:
    src = src.replace(
        "await self._harness.stop()",
        "await self._pool.stop()\n        await self._harness.stop()",
        1,
    )

target.write_text(src)
print("patched:", target)
# Print the diff-like lines we touched
print("---verify---")
for line in src.splitlines():
    if "WarmPool" in line or "_pool" in line:
        print("   ", line)
