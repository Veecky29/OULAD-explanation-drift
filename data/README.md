# Data directory (not committed)

Place downloaded datasets here. Raw files and feature caches are gitignored.

## OULAD (UCI Machine Learning Repository #349)

```bash
# from the repository root
mkdir -p data && cd data
curl -L -o oulad.zip "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
unzip -o oulad.zip
cd ..
```

Dataset page: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

OULAD is released under a Creative Commons Attribution 4.0 licence (Kuzilek et al., 2017). This repository does not redistribute the raw files; please download them from UCI.

The Open University site download link sometimes returns an HTML page rather than the archive; the UCI mirror is the copy used here (and by Wu et al., 2026). After unzipping you should have `studentVle.csv`, `studentInfo.csv`, `studentAssessment.csv`, `studentRegistration.csv`, `assessments.csv`, `courses.csv`, `vle.csv`.

Feature tables (`features_d{7,14,28,56,112}.parquet`) are built by `src/features.py` and are also gitignored.

## Riestra-González et al. (2021) Moodle data (external validation)

Clone the authors' public repository (Apache-2.0). `src/phase3_riestra.py` looks for the supervised tables in this order: the `RIESTRA_DATA` environment variable, `external/riestra/data/supervised` next to this repo, then `/workspace/external/riestra/data/supervised` (the path used on the original experiment machine).

```bash
# recommended: clone beside this repository
git clone https://github.com/moisesriestra/moodle-early-performance-prediction.git external/riestra

# or point the scripts at a checkout elsewhere
# export RIESTRA_DATA=/path/to/moodle-early-performance-prediction/data/supervised
```

The scripts read `data/supervised/clean_df_{cut}_5.0.csv` from that clone (semicolon-separated; cutoffs 10/25/33/50% of course length).
