# Chordonomicon Research Explorer

A local, non-commercial research application for exploring harmonic progression patterns in the Chordonomicon v2 corpus.

The app combines corpus search, next-chord statistics, transposition-invariant progression representations, and a supervised key-inference model that enables Roman-numeral analysis.

## Highlights

- Explore chord progressions by section, genre, subgenre, and decade.
- Compare absolute and transposition-invariant progression patterns.
- Estimate the next chord from corpus frequencies.
- Infer song key with a balanced HistGradientBoosting model.
- Display Roman-numeral interpretations from the inferred key.
- Run fully offline after local data/model setup.
- Query the corpus through DuckDB and Streamlit.

## Key-inference pipeline

The final v1 model uses:

- 24 candidate keys: 12 major + 12 minor
- candidate-relative, transposition-invariant features
- HistGradientBoosting candidate ranking
- inverse-frequency true-mode weighting (`alpha = 1.0`)
- temperature scaling (`T ≈ 1.3006`)

Post-hoc ChoCo test performance after model-selection work:

| Metric | Result |
|---|---:|
| Overall Top-1 key accuracy | 71.9% |
| Major Top-1 key accuracy | 72.4% |
| Minor Top-1 key accuracy | 69.0% |
| Balanced mode accuracy | 84.0% |

The earlier unbalanced model achieved slightly higher aggregate Top-1 accuracy but strongly underperformed on minor keys. The balanced model was selected because it substantially reduced major/minor asymmetry.

## Architecture

```text
Chordonomicon v2 CSV
        |
        v
preprocessing / section parsing
        |
        v
DuckDB aggregate tables
        |
        +----------------------+
        |                      |
        v                      v
progression explorer      key-inference model
        |                      |
        v                      v
next-chord statistics     calibrated key estimate
        |                      |
        +----------+-----------+
                   |
                   v
          Streamlit interface
                   |
                   v
           Roman-numeral view
```

## Local setup

The public repository intentionally does **not** include the raw Chordonomicon dataset, large DuckDB files, full prediction Parquet files, or trained model artifacts.

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the required local data/model artifacts under `data/` according to the project documentation.

Then run:

```bash
./run.command
```

or:

```bash
streamlit run app.py
```

## Data and licensing

Chordonomicon v2 is a third-party dataset distributed under **CC BY-NC 4.0**. This repository contains application and analysis code only; the dataset itself is not redistributed here.

This project is intended as a personal/non-commercial research and portfolio demonstration. Dataset licensing and any underlying third-party rights should be reviewed separately before commercial reuse.

## Methodological notes

- Relative-root representation is not the same as Roman-numeral analysis.
- Roman numerals are generated only after a key has been inferred.
- Model probabilities were calibrated on ChoCo. Calibration can shift when predictions are applied to Chordonomicon.
- Chordonomicon chord sequences should not be interpreted as duration-aware harmonic rhythm.
- Section-level modulation detection is outside v1.

## Repository structure

```text
.
├── app.py
├── chord_utils.py
├── db.py
├── key_modeling/
├── scripts/
├── tests/
├── docs/
├── requirements.txt
├── run.command
└── data/                 # local-only, ignored by Git
```

## Why this project

The project started as a corpus-based progression explorer and evolved into an end-to-end analytical application: data preprocessing, SQL/DuckDB aggregation, feature engineering, supervised learning, imbalance diagnostics, calibration, deployment-domain auditing, and interactive visualization.

It is particularly useful as a portfolio example of applied data science because the modeling choices were driven by diagnostic evidence rather than aggregate accuracy alone.
