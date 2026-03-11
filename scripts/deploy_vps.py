"""
scripts/deploy_vps.py — Upload changed files to VPS and restart the bot.
Run from project root: python scripts/deploy_vps.py
"""
import os
import time
import paramiko

LOCAL_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_ROOT = "/home/administrator/Super Trade Coin/crypto-ai-trading-system"
HOST = "77.93.155.115"
USER = "administrator"
PASS = "Abc@1234"

FILES = [
    "config.py",
    "dashboard/api_server.py",
    "dashboard/dashboard.html",
    "scanners/market_scanner.py",
    "scanners/research_engine.py",
    "trading/risk_manager.py",
    "data_engine/coin_database.py",
    "ai_engine/__init__.py",
    "ai_engine/llm_analyzer.py",
]

log_lines = []


def run(ssh, cmd):
    _, out, err = ssh.exec_command(cmd)
    exit_code = out.channel.recv_exit_status()
    return out.read().decode("utf-8", errors="replace").strip(), exit_code


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=22, username=USER, password=PASS, timeout=15)
    log_lines.append("SSH connected to " + HOST)

    # Ensure ai_engine directory exists on VPS
    run(ssh, f'mkdir -p "{REMOTE_ROOT}/ai_engine"')
    log_lines.append("ai_engine/ dir ensured on VPS")

    # Upload files via SFTP
    sftp = ssh.open_sftp()
    for rel_path in FILES:
        local_path  = os.path.join(LOCAL_ROOT, rel_path.replace("/", os.sep))
        remote_path = REMOTE_ROOT + "/" + rel_path
        sftp.put(local_path, remote_path)
        log_lines.append("uploaded: " + rel_path)
    sftp.close()
    log_lines.append("All files uploaded")

    # Restart service (SIGKILL to avoid 45s wait)
    restart_cmd = (
        f"echo '{PASS}' | sudo -S systemctl kill -s SIGKILL super-trade-coin "
        f"&& sleep 2 "
        f"&& echo '{PASS}' | sudo -S systemctl start super-trade-coin"
    )
    _, rc = run(ssh, restart_cmd)
    log_lines.append(f"restart command exit_code={rc}")

    # Wait for service to come up
    time.sleep(5)
    status, _ = run(ssh, f"echo '{PASS}' | sudo -S systemctl is-active super-trade-coin")
    log_lines.append("service status: " + status)

    # Grab last journal lines
    journal, _ = run(
        ssh,
        f"echo '{PASS}' | sudo -S journalctl -u super-trade-coin -n 15 --no-pager 2>&1",
    )

    ssh.close()

    # Write results to file (avoid Windows console encoding issues)
    result_path = os.path.join(LOCAL_ROOT, "deploy_result.txt")
    with open(result_path, "w", encoding="utf-8") as fh:
        fh.write("=== DEPLOY LOG ===\n")
        for line in log_lines:
            fh.write(line + "\n")
        fh.write("\n=== JOURNAL (last 15 lines) ===\n")
        fh.write(journal + "\n")

    print("Deploy finished. Status:", status)
    print("Full log saved to deploy_result.txt")
    print("\n--- JOURNAL ---")
    # Print safely for Windows cp1252
    safe = journal.encode("cp1252", errors="replace").decode("cp1252")
    print(safe)


if __name__ == "__main__":
    main()
