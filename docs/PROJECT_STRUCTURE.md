# Recommended local project structure

After cleanup, keep the working folder approximately like this:

```text
chordonomicon_research_explorer/
├── app.py
├── chord_utils.py
├── db.py
├── requirements.txt
├── run.command
├── README.md
├── .gitignore
│
├── key_modeling/
│   ├── key_inference.py
│   ├── ml_features.py
│   ├── predict_key_ml.py
│   └── ...
│
├── scripts/
│   └── organize_project.py
│
├── tests/
│
├── data/
│   ├── chordonomicon_v2.csv
│   ├── chordonomicon.duckdb
│   └── key_models/
│       ├── chordonomicon_song_keys_mode_balanced.parquet
│       ├── chordonomicon_song_keys_mode_balanced_summary.json
│       └── mode_balance_experiment/
│           ├── selected_mode_balanced_histgb.joblib
│           └── mode_balanced_temperature_calibrator.json
│
└── _archive/
    ├── patches/
    ├── experiment_reports/
    └── old_outputs/
```

`data/` remains local and is excluded from GitHub.
`_archive/` contains historical development artifacts and is also excluded from GitHub.
