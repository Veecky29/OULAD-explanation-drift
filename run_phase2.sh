#!/usr/bin/env bash
# Phase 2 (CPU; ~1-2 h with 3 parallel workers). Requires features for cutoffs 7 14 28 56 112.
set -e
cd "$(dirname "$0")"; . .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for c in 7 14 28 56 112; do [ -f data/features_d$c.parquet ] || python src/features.py --cutoff $c; done
printf "28 base\n56 base\n7 base\n14 base\n112 base\n28 novisits\n" | \
  xargs -P 3 -L 1 sh -c 'python -u src/phase2.py --cutoff $0 --variant $1 --B 30 --jobs 2 > results/phase2/log_d$0_$1.txt 2>&1'
for c in 28 56; do python src/phase2_null.py --cutoff $c > results/phase2/log_null_d$c.txt 2>&1; done
python src/phase2_analysis.py
