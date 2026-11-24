#!/usr/bin/env bash
# Phase 3 (CPU). Per-pair checkpointing: re-running skips finished jobs.
# Threads capped per process to avoid oversubscription (phase-2 lesson).
set -e
cd "$(dirname "$0")"; . .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKERS=${WORKERS:-5}
mkdir -p results/phase3/logs
# 1) B=30 null controls for the remaining cutoffs (phase-2 design)
for c in 7 14 112; do [ -f results/phase2/null_d$c/pairs.csv ] && grep -q "" results/phase2/null_d$c/pairs.csv && [ $(wc -l < results/phase2/null_d$c/pairs.csv) -ge 15 ] || python -u src/phase2_null.py --cutoff $c > results/phase2/log_null_d$c.txt 2>&1; done
# 2) B=100 source+target bootstrap jobs; d28/d56 first, then d7/d14/d112
python src/phase3_jobs.py > results/phase3/jobs.txt
python src/phase3_riestra_jobs.py > results/phase3/riestra_jobs.txt
xargs -P $WORKERS -L 1 -a results/phase3/jobs.txt sh -c 'python -u src/phase3.py --cutoff $0 --model $1 --kind $2 --pair "$3 $4" --B 100 >> results/phase3/logs/jobs.log 2>&1 || echo "FAIL $0 $1 $2 $3 $4" >> results/phase3/logs/jobs.log'
# 3) external replication (Riestra) and analysis
python src/phase3_trust_oulad.py > results/phase3/logs/trust_oulad.log 2>&1
xargs -P $WORKERS -L 1 -a results/phase3/riestra_jobs.txt sh -c 'python -u src/phase3_riestra.py $0 $1 $2 $3 >> results/phase3/logs/riestra.log 2>&1'
python src/phase3_analysis.py > results/phase3/logs/analysis.log 2>&1
python src/phase3_md_tables.py > results/phase3/phase3_tables.md
