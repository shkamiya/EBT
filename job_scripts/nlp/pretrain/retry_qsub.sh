#!/bin/bash
# retry_qsub.sh
# qsub を N 回、60 秒ごとに実行するだけの簡易スクリプト

JOB_SCRIPT="job_scripts/nlp/pretrain/ebt_s1_miyabi.sh"   # ←送信したい PBS スクリプト
MAX_TRIES=100        # 何回まで試すか。0 なら無限
SLEEP_SEC=60         # インターバル（秒）

i=0
while [[ $MAX_TRIES -eq 0 || $i -lt $MAX_TRIES ]]; do
    echo "[$(date '+%F %T')] try $((i+1)) : qsub $JOB_SCRIPT"
    qsub "$JOB_SCRIPT"
    ((i++))
    sleep "$SLEEP_SEC"
done
