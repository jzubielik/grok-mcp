"""Run inside the grok-mcp container as the server's uid (`make verify-secrets`): fail if any
/proc/<pid>/environ readable from there holds a secret. tini, as PID 1, gets Docker's whole
environment, so this catches a deployment where it runs under the agent's uid.

Its own process is skipped: `docker exec` hands it the container environment too. Started as
root (the compose default user), it first drops to GROK_RUN_UID:GROK_RUN_GID like the server —
the kernel opens another process's environ only to a matching uid AND gid.
"""

import os
import re
import sys

SECRET = re.compile(rb"^(XAI_API_KEY|MCP_AUTH_TOKEN|PGPASSWORD|POSTGRES_PASSWORD|DATABASE_URL)=")

if os.getuid() == 0:
    os.setgroups([])
    os.setgid(int(os.environ["GROK_RUN_GID"]))
    os.setuid(int(os.environ["GROK_RUN_UID"]))

leaks = []
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == os.getpid():
        continue
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            env = f.read()
    except OSError:
        continue
    leaks += [(pid, v.split(b"=")[0].decode()) for v in env.split(b"\0") if SECRET.match(v)]

for pid, name in leaks:
    print(f"LEAK: {name} readable in /proc/{pid}/environ")
if not leaks:
    print("ok: no secret readable through /proc/*/environ")
sys.exit(1 if leaks else 0)
