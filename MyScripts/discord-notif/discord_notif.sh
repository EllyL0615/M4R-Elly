#!/bin/bash

# Usage: 
# 1. save Discord channel URL to bashrc (One-time)
#     $ echo 'export DISCORD_WEBHOOK_URL="https://discordapp.com/api/webhooks/xxx/yyy"' >> ~/.bashrc
# 2. source AFTER conda activation in PBS scripts
#     source ~/.bashrc
#     source /rds/general/user/yl9422/home/files/M4R-Elly/MyScripts/discord-notif/discord_notif.sh

# If not running under PBS, exit quietly
if [ -z "${PBS_JOBID}" ]; then
    return 0 2>/dev/null || exit 0
fi

# function definition
send_discord_notif() {
    local STATUS="$1"
    local JOB_NAME=${PBS_JOBNAME:-"unknown_job"}
    local JOB_ID=${PBS_JOBID:-"unknown_id"}
    local TIME
    TIME=$(date "+%Y-%m-%d %H:%M:%S")

    local MESSAGE="**CX3 Batch Job Update** 🚀
- **Job Name:** \`${JOB_NAME}\`
- **Job ID:** \`${JOB_ID}\`
- **Status:** **${STATUS}**
- **Time:** ${TIME}"

    python - << EOF
import requests

url = "${DISCORD_WEBHOOK_URL}"
data = {"content": """${MESSAGE}"""}

try:
    requests.post(url, json=data, timeout=10)
except Exception:
    traceback.print_exc()
EOF
}

send_discord_notif "STARTED"

on_exit() {
    local EXIT_CODE=$?
    if [ ${EXIT_CODE} -eq 0 ]; then
        send_discord_notif "FINISHED"
    else
        send_discord_notif "ABORTED (exit code = ${EXIT_CODE})"
    fi
}

trap 'on_exit' EXIT