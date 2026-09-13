# GitHub portfolio checklist

1. Clean the local project without deleting anything:
   ```bash
   python scripts/organize_project.py
   python scripts/organize_project.py --apply
   ```

2. Replace/merge the old README with `README_GITHUB.md`.

3. Initialize Git:
   ```bash
   git init
   git add .
   git status
   ```

4. Check carefully that no CSV, Parquet, DuckDB, or `.joblib` files are staged.

5. Commit:
   ```bash
   git commit -m "Initial public portfolio release"
   ```

6. Create a GitHub repository, then:
   ```bash
   git branch -M main
   git remote add origin <YOUR_REPOSITORY_URL>
   git push -u origin main
   ```

## Good portfolio additions

- Add 2–4 screenshots under `docs/screenshots/`.
- Put one screenshot near the top of the README.
- Add a short "Methods" section explaining the imbalance diagnosis and why the balanced model was preferred.
- Add a small architecture diagram.
- Add a 30–60 second screen recording or GIF if desired.

## Do not upload

- `chordonomicon_v2.csv`
- `.duckdb` files
- full `.parquet` prediction files
- large trained model artifacts unless you have separately reviewed redistribution implications
- any third-party dataset content
