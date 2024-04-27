#!/bin/bash
if [ -z "$2" ]; then
    jobname=$1
else
    jobname=$2
fi

sbatch -a 1-$(cat $1 | wc -l) -p cpu -J $jobname \
       -c 1 --mem=12g -t 40:00 \
       --wrap="eval \`sed -n \${SLURM_ARRAY_TASK_ID}p $1\`" \
       #--gres=gpu:a4000:1 \
       #--dependency=afterok:25345214
       #--nice=1 \
       #-o /dev/null -e /dev/null \
