#!/bin/bash
# Chain dev_accelerated (1h) jobs until an accelerated job starts running.
# Each slot checks at startup: if ACCEL_JOB is RUNNING, exit immediately.
# Resume logic in the inference script ensures no work is repeated.
#
# Usage: bash submit_dev_chain.sh <accel_job_id> <sbatch_script> [n_slots]
# Example: bash submit_dev_chain.sh 4058434 sbatch_inference_vkitti_baseline_radius.sh 30

ACCEL_JOB=${1:?Usage: $0 <accel_job_id> <sbatch_script> [n_slots]}
SCRIPT=${2:?Usage: $0 <accel_job_id> <sbatch_script> [n_slots]}
N=${3:-30}

PREV=""
for i in $(seq 1 $N); do
    DEP_FLAG=""
    [ -n "$PREV" ] && DEP_FLAG="--dependency=afterok:$PREV"

    JID=$(sed 's/00:10:00/01:00:00/' "$SCRIPT" | \
          sbatch --parsable $DEP_FLAG \
          --export=ALL,ACCEL_JOB=$ACCEL_JOB,HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN)

    if [ -z "$JID" ]; then
        echo "Submission failed at slot $i (QOS limit?). Stopping at $((i-1)) slots."
        break
    fi

    echo "Chain slot $i: job $JID (depends on: ${PREV:-none})"
    PREV=$JID
done

echo "Done. The last job in the chain will self-resubmit until job $ACCEL_JOB starts running."
