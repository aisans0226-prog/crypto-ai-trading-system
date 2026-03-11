"""
updater/auto_updater.py - VPS Auto-Update System.

Workflow:
  1. Poll GitHub for new commits on the configured branch
  2. If new version found: pull, install deps, graceful restart
  3. Pre-update backup of config files
  4. POST /api/update/apply triggers manual update
  5. POST /api/update/rollback rolls back to previous commit
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List

from loguru import logger

BASE_DIR = Path(__file__).parent.parent.resolve()
BACKUP_DIR = BASE_DIR / ".backups"
BRANCH = "master"
CHECK_INTERVAL = 3600


def _run(cmd: List[str], cwd: Path = BASE_DIR):
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=120)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


class AutoUpdater:
    def __init__(self) -> None:
        self._current_commit: str = self._get_current_commit()
        self._update_available: bool = False
        self._latest_commit: str = ""
        self._update_history: List[dict] = []
        self._running: bool = False
        BACKUP_DIR.mkdir(exist_ok=True)

    def _get_current_commit(self) -> str:
        rc, out, _ = _run(["git", "rev-parse", "HEAD"])
        return out[:7] if rc == 0 else "unknown"

    def _get_remote_commit(self) -> str:
        _run(["git", "fetch", "origin", BRANCH])
        rc, out, _ = _run(["git", "rev-parse", f"origin/{BRANCH}"])
        return out[:7] if rc == 0 else ""

    async def check_for_update(self) -> dict:
        loop = asyncio.get_running_loop()
        remote = await loop.run_in_executor(None, self._get_remote_commit)
        self._latest_commit = remote
        self._update_available = bool(remote and remote != self._current_commit[:7])
        return {
            "current_commit": self._current_commit,
            "latest_commit": remote,
            "update_available": self._update_available,
        }

    async def apply_update(self) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._do_update)

    def _do_update(self) -> dict:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        result = {"timestamp": ts, "previous_commit": self._current_commit, "success": False, "message": ""}
        try:
            self._backup_config(ts)
            rc, out, err = _run(["git", "pull", "origin", BRANCH])
            if rc != 0:
                result["message"] = f"git pull failed: {err}"
                return result
            _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
            new_commit = self._get_current_commit()
            result.update({"success": True, "new_commit": new_commit,
                           "message": f"Updated {self._current_commit} -> {new_commit}"})
            self._update_history.append(result)
            self._current_commit = new_commit
            self._update_available = False
            logger.info("Update applied: {}", result["message"])
            threading.Timer(3, self._restart).start()   # delayed restart (thread-safe)
        except Exception as exc:
            result["message"] = str(exc)
            logger.error("Update failed: {}", exc)
        return result

    def _backup_config(self, ts: str) -> None:
        backup = BACKUP_DIR / ts
        backup.mkdir(parents=True, exist_ok=True)
        for name in [".env", "config.py"]:
            src = BASE_DIR / name
            if src.exists():
                shutil.copy2(str(src), str(backup / name))

    @staticmethod
    def _restart() -> None:
        logger.info("Restarting process to apply update...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    async def rollback(self) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._do_rollback)

    def _do_rollback(self) -> dict:
        rc, out, err = _run(["git", "log", "--oneline", "-2"])
        lines = out.splitlines()
        if len(lines) < 2:
            return {"success": False, "message": "No previous commit to roll back to"}
        prev_hash = lines[1].split()[0]
        rc2, _, err2 = _run(["git", "reset", "--hard", prev_hash])
        if rc2 != 0:
            return {"success": False, "message": err2}
        self._current_commit = self._get_current_commit()
        threading.Timer(3, self._restart).start()    # delayed restart (thread-safe)
        return {"success": True, "commit": prev_hash, "message": f"Rolled back to {prev_hash}"}

    async def start_polling(self) -> None:
        self._running = True
        while self._running:
            try:
                status = await self.check_for_update()
                if status["update_available"]:
                    logger.info("Update available: {} -> {}", status["current_commit"], status["latest_commit"])
            except Exception as exc:
                logger.debug("Update check error: {}", exc)
            await asyncio.sleep(CHECK_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    def get_status(self) -> dict:
        return {
            "current_commit": self._current_commit,
            "latest_commit": self._latest_commit,
            "update_available": self._update_available,
            "recent_history": self._update_history[-5:],
            "branch": BRANCH,
        }
