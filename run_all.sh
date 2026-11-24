#!/usr/bin/env bash
# Reproduce the pilot end-to-end (CPU only; ~45 min on 8 cores).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
mkdir -p data && cd data
[ -f studentVle.csv ] || { curl -L -o oulad.zip "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip" && unzip -o oulad.zip; }
cd ..
for c in 28 56; do
  python src/features.py --cutoff $c
  python src/experiment.py --cutoff $c
  python src/figures.py --cutoff $c
  python src/noise_floor.py --cutoff $c
done
python src/leakage_checks.py
