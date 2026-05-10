# UCDDB Apnea Classification

Minimal Python project for a master's thesis on binary apnea/hypopnea classification using the UCDDB dataset.

At this stage, the project only implements the first step: raw data audit.

## Project Structure

```text
ucddb-apnea/
+-- data/
|   +-- raw/
|   +-- processed/
+-- scripts/
|   +-- 00_download_ucddb.py
|   +-- 01_data_audit.py
|   +-- 02_build_epochs.py
|   +-- 03_build_advanced_features.py
|   +-- 03_build_model_ready_features.py
|   +-- 03_extract_features.py
|   +-- 03_validate_dataset.py
|   +-- 03_build_segment_features.py
|   +-- 04_diagnose_results.py
|   +-- 04_train_improved_models.py
|   +-- 04_train_models.py
|   +-- 04_train_segment_models_reduced.py
|   +-- 04_train_temporal_ensemble.py
|   +-- 05_analyze_results.py
|   +-- 06_interpret_final_models.py
|   +-- 06_audit_pipeline.py
|   +-- 07_make_final_artifacts.py
+-- reports/
|   +-- figures/
|   +-- tables/
+-- config.py
+-- README.md
+-- requirements.txt
+-- .gitignore
```

## Data

Place the downloaded UCDDB files into:

```text
data/raw/
```

For the PhysioNet UCDDB release, the downloader keeps only files needed for this project:

```text
RECORDS
SubjectDetails.xls
SHA256SUMS.txt
ucddb*.rec
ucddb*_respevt.txt
ucddb*_stage.txt
```

Large `*_lifecard.edf` files are intentionally not downloaded.

Respiratory event annotations are expected in files such as:

```text
ucddb002_respevt.txt
```

The `data/raw/` directory is ignored by Git.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Download UCDDB Files

From the project root:

```bash
python scripts/00_download_ucddb.py
```

The script downloads the selected UCDDB files from PhysioNet into `data/raw/`.
Existing complete files are skipped after checking their remote size.

## Run Data Audit

From the project root:

```bash
python scripts/01_data_audit.py
```

The script scans `data/raw/`, tries to read each UCDDB EDF signal file, prints channel names and sampling frequencies, checks respiratory event annotation files, and saves the audit table to:

```text
reports/tables/ucddb_audit.csv
```

## Build Epoch Table

From the project root:

```bash
python scripts/02_build_epochs.py
```

The script converts UCDDB records into 30-second epochs with binary labels:
`normal` and `apnea_hypopnea`. The main output is:

```text
data/processed/epochs.csv
```

## Extract Features

From the project root:

```bash
python scripts/03_extract_features.py
```

The script extracts ECG, Flow, ribcage, abdominal effort, SpO2, and combined effort features for each 30-second epoch. The main output is:

```text
data/processed/features_all.csv
```

## Build Model-Ready Features

After extracting the base feature table, run:

```bash
python scripts/03_build_model_ready_features.py
```

The script keeps all original columns and adds sleep filtering, record-normalized, and neighboring-epoch context columns. The main output is:

```text
data/processed/features_model_ready.csv
```

## Build Advanced Features

To add wider temporal context and interaction features, run:

```bash
python scripts/03_build_advanced_features.py
```

The script creates:

```text
data/processed/features_advanced.csv
```

## Validate Prepared Dataset

Before training models, run:

```bash
python scripts/03_validate_dataset.py
```

The script checks table consistency, labels, epoch timing, missing values, feature ranges, and subject-level split readiness. The main report is:

```text
reports/validation_report.md
```

## Train Models

After feature extraction and validation, run:

```bash
python scripts/04_train_models.py
```

The script trains subject-level cross-validation models and saves result tables to `reports/tables/`.

## Train Improved Models

After building `features_model_ready.csv`, run:

```bash
python scripts/04_train_improved_models.py
```

The script compares all-epoch and sleep-only regimes, base and enhanced features, and fusion strategies.

## Train Advanced Models

After building `features_advanced.csv`, run:

```bash
python scripts/04_train_advanced_models.py
```

The script tests advanced temporal/context features and six XGBoost configurations using subject-level cross-validation.

## Train Reduced Segment-Level Models

After building `data/processed/segment_features.csv`, run:

```bash
python scripts/04_train_segment_models_reduced.py
```

The script runs a lighter honest segment-level experiment with subject-level cross-validation and saves only aggregated results, thresholds, and a markdown report.

## Train Temporal Ensemble

After building `features_model_ready.csv`, run:

```bash
python scripts/04_train_temporal_ensemble.py
```

The script compares the best SpO2 baseline with a small Flow+SpO2 ensemble and label-free temporal probability smoothing under subject-level validation.

## Interpret Final Models

After building `features_model_ready.csv`, run:

```bash
python scripts/06_interpret_final_models.py
```

The script trains final component models on sleep-only data for feature importance analysis and saves interpretability tables, figures, and a markdown report.

## Diagnose Baseline Results

After training baseline models, run:

```bash
python scripts/04_diagnose_results.py
```

The script analyzes fold balance, sleep-stage label distribution, threshold sensitivity, and top baseline experiments without retraining models.

## Analyze Final Results

After training improved models, run:

```bash
python scripts/05_analyze_results.py
```

The script creates final thesis tables, threshold analysis, figures, and a markdown report from saved improved-model predictions.

## Make Final Thesis Artifacts

After the model reports have been generated, run:

```bash
python scripts/07_make_final_artifacts.py
```

The script collects the final compact tables, figures, and artifact index for writing the thesis.
