"""
scripts/auto_monitor.py — Automatic VPS log monitor + AI-powered error fixer.

Algorithm every CHECK_INTERVAL seconds:
  1. Pull recent journal lines from VPS via SSH
  2. Detect real ERROR/CRITICAL/Traceback events (filters noise)
  3. If new error found:
     a. Extract traceback + map VPS path → local file
     b. Read local file, call Claude API → get fixed code
     c. Backup original, write fix, deploy only that file
     d. Wait 30s, verify service still running
     e. If still crashing → revert backup + send alert
     f. Telegram notification on success or failure

Usage:
    python scripts/auto_monitor.py               # run once & exit
    python scripts/auto_monitor.py --watch       # continuous loop (default 5 min)
    python scripts/auto_monitor.py --watch --interval 120
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import paramiko

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BACKUP_DIR  = PROJECT_DIR / ".fix_backups"
STATE_FILE  = SCRIPT_DIR / "auto_monitor_state.json"
LOG_FILE    = SCRIPT_DIR / "auto_monitor.log"
BACKUP_DIR.mkdir(exist_ok=True)

# ── VPS config (mirrors deploy_vps.py) ───────────────────────────────────────
VPS_HOST    = "77.93.155.115"
VPS_USER    = "administrator"
VPS_PASS    = "Abc@1234"
VPS_ROOT    = "/home/administrator/Super Trade Coin/crypto-ai-trading-system"
SERVICE     = "super-trade-coin"

# ── Check interval ────────────────────────────────────────────────────────────
CHECK_INTERVAL    = 300    # seconds (5 min)
LOG_LINES_TO_PULL = 120    # journal lines per check
FIX_COOLDOWN_MIN  = 10     # don't fix same error fingerprint within N minutes
MAX_FIX_ATTEMPTS  = 2      # give up and alert after this many failed fix attempts

# ── Noise patterns to ignore ──────────────────────────────────────────────────
NOISE_PATTERNS = [
    r"cuDNN|cuFFT|cuBLAS|TF-TRT|TensorRT|tensorflow|AVX2|cpu_feature_guard",
    r"\[sudo\] password",
    r"systemd\[1\]:",       # systemd lifecycle messages handled separately
    r"WARNING.*aiohttp",
    r"WARNING.*asyncio",
]

# ── Real-error patterns ───────────────────────────────────────────────────────
ERROR_KEYWORDS = [
    r"\| ERROR \|",
    r"\| CRITICAL \|",
    r"Traceback \(most recent call last\)",
    r"(AttributeError|TypeError|KeyError|NameError|ImportError|ModuleNotFoundError"
    r"|ValueError|RuntimeError|SyntaxError|IndentationError|AssertionError):",
]

# ── VPS path → local path mapping ────────────────────────────────────────────
def vps_to_local(vps_path: str) -> Optional[Path]:
    prefix = VPS_ROOT + "/"
    if vps_path.startswith(prefix):
        rel = vps_path[len(prefix):]
        local = PROJECT_DIR / rel.replace("/", os.sep)
        return local if local.exists() else None
    return None


# ── State persistence ─────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_check_ts": None, "fix_history": {}}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))

def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    safe = line.encode("ascii", errors="replace").decode("ascii")
    print(safe)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── SSH helpers ───────────────────────────────────────────────────────────────
def ssh_connect() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_HOST, port=22, username=VPS_USER, password=VPS_PASS, timeout=15)
    return ssh

def ssh_run(ssh: paramiko.SSHClient, cmd: str) -> str:
    _, out, err = ssh.exec_command(cmd)
    out.channel.recv_exit_status()
    return out.read().decode("utf-8", errors="replace")

def get_vps_logs(n_lines: int = LOG_LINES_TO_PULL) -> str:
    ssh = ssh_connect()
    try:
        return ssh_run(ssh,
            f"echo '{VPS_PASS}' | sudo -S journalctl -u {SERVICE} "
            f"-n {n_lines} --no-pager 2>&1"
        )
    finally:
        ssh.close()

def get_service_status() -> str:
    ssh = ssh_connect()
    try:
        return ssh_run(ssh,
            f"echo '{VPS_PASS}' | sudo -S systemctl is-active {SERVICE} 2>&1"
        ).strip()
    finally:
        ssh.close()

def restart_service(ssh: paramiko.SSHClient) -> None:
    ssh_run(ssh,
        f"echo '{VPS_PASS}' | sudo -S systemctl kill -s SIGKILL {SERVICE} "
        f"&& sleep 2 "
        f"&& echo '{VPS_PASS}' | sudo -S systemctl start {SERVICE}"
    )


# ── Log parsing ───────────────────────────────────────────────────────────────
def is_noise(line: str) -> bool:
    return any(re.search(p, line) for p in NOISE_PATTERNS)

def extract_errors(raw_log: str) -> list[dict]:
    """Return list of {fingerprint, traceback_text, file_path, line_no}."""
    lines = raw_log.splitlines()
    errors = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if is_noise(line):
            i += 1
            continue

        # Detect start of a real error block
        if not any(re.search(p, line) for p in ERROR_KEYWORDS):
            i += 1
            continue

        # Collect the full block (up to 20 lines)
        block = []
        j = i
        while j < len(lines) and j < i + 25:
            if not is_noise(lines[j]):
                block.append(lines[j])
            j += 1

        block_text = "\n".join(block)

        # Extract first "File ...line N" reference (VPS path)
        file_path = None
        line_no   = None
        for b in block:
            m = re.search(r'File "([^"]+)", line (\d+)', b)
            if m:
                file_path = m.group(1)
                line_no   = int(m.group(2))

        # Fingerprint = hash of the first error line (stable across restarts)
        fingerprint = hashlib.md5(block[0].encode()).hexdigest()[:12]

        errors.append({
            "fingerprint":    fingerprint,
            "traceback_text": block_text,
            "file_path":      file_path,
            "line_no":        line_no,
        })
        i = j  # skip past this block

    return errors


# ── AI fix via Anthropic API ──────────────────────────────────────────────────
def _load_api_key() -> Optional[str]:
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val and val != "your_key_here":
                    return val
    return os.environ.get("ANTHROPIC_API_KEY")

def ai_fix_file(file_content: str, filename: str, traceback_text: str) -> Optional[str]:
    """Call Claude API with (file + error) → return fixed file content or None."""
    api_key = _load_api_key()
    if not api_key:
        log("  ! No ANTHROPIC_API_KEY found — cannot auto-fix")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            f"You are fixing a Python bug in a crypto trading bot.\n\n"
            f"ERROR LOG:\n```\n{traceback_text[:3000]}\n```\n\n"
            f"FILE: {filename}\n```python\n{file_content[:6000]}\n```\n\n"
            f"Instructions:\n"
            f"- Return ONLY the complete fixed Python file content\n"
            f"- No explanations, no markdown fences, just the raw code\n"
            f"- Keep all existing logic intact; only fix the specific bug\n"
            f"- If you cannot determine a safe fix, reply with exactly: CANNOT_FIX"
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text.strip()
        if result == "CANNOT_FIX" or not result.startswith(("import ", "\"\"\"", "#", "from ", "class ", "async ", "def ")):
            log(f"  ! AI returned CANNOT_FIX or invalid content for {filename}")
            return None
        return result
    except Exception as exc:
        log(f"  ! AI fix call failed: {exc}")
        return None


# ── Deploy single file ────────────────────────────────────────────────────────
def deploy_file(local_path: Path, rel_path: str) -> bool:
    try:
        ssh = ssh_connect()
        sftp = ssh.open_sftp()
        remote = VPS_ROOT + "/" + rel_path
        sftp.put(str(local_path), remote)
        sftp.close()
        restart_service(ssh)
        ssh.close()
        log(f"  + Deployed {rel_path} and restarted service")
        return True
    except Exception as exc:
        log(f"  ! Deploy failed: {exc}")
        return False


# ── Telegram alert ────────────────────────────────────────────────────────────
def _load_telegram_config() -> tuple[Optional[str], Optional[str]]:
    env_file = PROJECT_DIR / ".env"
    token, chat_id = None, None
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"')
            if line.startswith("TELEGRAM_CHAT_ID="):
                chat_id = line.split("=", 1)[1].strip().strip('"')
    return token, chat_id

def send_telegram(msg: str) -> None:
    token, chat_id = _load_telegram_config()
    if not token or not chat_id:
        return
    try:
        import urllib.request, urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode()
        urllib.request.urlopen(url, data, timeout=5)
    except Exception:
        pass


# ── Main fix flow ─────────────────────────────────────────────────────────────
def handle_error(error: dict, state: dict) -> bool:
    fp = error["fingerprint"]
    history = state["fix_history"]

    # Check cooldown — don't re-fix same error within FIX_COOLDOWN_MIN
    if fp in history:
        last_attempt = datetime.fromisoformat(history[fp]["last_attempt"])
        attempts     = history[fp].get("attempts", 0)
        if datetime.utcnow() - last_attempt < timedelta(minutes=FIX_COOLDOWN_MIN):
            log(f"  ~ Skipping {fp}: in cooldown ({FIX_COOLDOWN_MIN}m)")
            return False
        if attempts >= MAX_FIX_ATTEMPTS:
            log(f"  ~ Skipping {fp}: max attempts ({MAX_FIX_ATTEMPTS}) reached")
            return False

    vps_file_path = error["file_path"]
    log(f"  >> Error fingerprint: {fp}")
    log(f"     VPS file: {vps_file_path}")
    log(f"     Traceback:\n{error['traceback_text'][:400]}")

    # Record attempt
    history.setdefault(fp, {"attempts": 0})
    history[fp]["last_attempt"] = datetime.utcnow().isoformat()
    history[fp]["attempts"]     += 1
    save_state(state)

    # Map to local file
    local_path = None
    rel_path   = None
    if vps_file_path:
        local_path = vps_to_local(vps_file_path)
        if local_path:
            rel_path = str(local_path.relative_to(PROJECT_DIR)).replace(os.sep, "/")

    if not local_path:
        msg = f"\U0001f534 <b>Bot Error (unfixable)</b>\nFile not in local codebase\n<pre>{error['traceback_text'][:800]}</pre>"
        send_telegram(msg)
        log("  ! File not in local codebase -- alert sent")
        return False

    # Read local file
    original_content = local_path.read_text(encoding="utf-8")

    # Call AI for fix
    log(f"  * Calling Claude API to fix {rel_path}...")
    fixed_content = ai_fix_file(original_content, rel_path, error["traceback_text"])

    if not fixed_content:
        msg = f"\U0001f534 <b>Bot Error (AI fix failed)</b>\nFile: <code>{rel_path}</code>\n<pre>{error['traceback_text'][:600]}</pre>"
        send_telegram(msg)
        return False

    # Backup original
    backup_path = BACKUP_DIR / f"{rel_path.replace('/', '_')}_{fp}.bak"
    shutil.copy2(local_path, backup_path)
    log(f"  + Backed up to {backup_path}")

    # Write fix
    local_path.write_text(fixed_content, encoding="utf-8")
    log(f"  + Fix written to {local_path}")

    # Deploy
    if not deploy_file(local_path, rel_path):
        # Revert on deploy failure
        shutil.copy2(backup_path, local_path)
        log("  ! Deploy failed — reverted")
        return False

    # Wait and verify
    log("  * Waiting 30s for service to stabilise...")
    time.sleep(30)
    status = get_service_status()
    log(f"  * Service status after fix: {status}")

    if status == "active":
        history[fp]["fixed"] = True
        save_state(state)
        msg = (f"\u2705 <b>Auto-Fixed</b>\nFile: <code>{rel_path}</code>\n"
               f"Error: <code>{fp}</code>\nService: active")
        send_telegram(msg)
        log(f"  [OK] Fix successful for {rel_path}")
        return True
    else:
        # Revert
        shutil.copy2(backup_path, local_path)
        deploy_file(local_path, rel_path)
        msg = (f"\U0001f534 <b>Auto-Fix Failed -- Reverted</b>\nFile: <code>{rel_path}</code>\n"
               f"<pre>{error['traceback_text'][:400]}</pre>")
        send_telegram(msg)
        log(f"  [FAIL] Fix did not resolve issue -- reverted {rel_path}")
        return False


# ── Monitor cycle ─────────────────────────────────────────────────────────────
def check_once(state: dict) -> None:
    log("=== Check cycle ===========================================")
    try:
        raw = get_vps_logs()
    except Exception as exc:
        log(f"  ! Cannot reach VPS: {exc}")
        return

    errors = extract_errors(raw)
    if not errors:
        log("  OK  No errors found")
    else:
        log(f"  !! {len(errors)} error(s) found")
        for err in errors:
            handle_error(err, state)

    state["last_check_ts"] = datetime.utcnow().isoformat()
    save_state(state)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Auto-monitor VPS bot logs")
    parser.add_argument("--watch",    action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL, metavar="SEC")
    args = parser.parse_args()

    state = load_state()
    log(f"Auto-monitor started | watch={args.watch} | interval={args.interval}s")

    if args.watch:
        while True:
            check_once(state)
            log(f"  ... sleeping {args.interval}s")
            time.sleep(args.interval)
    else:
        check_once(state)


if __name__ == "__main__":
    main()
