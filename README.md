# UCDDB Apnea Classification

Репозиторий к ВКР по бинарной классификации эпизодов апноэ/гипопноэ на датасете UCDDB. Единица анализа - 30-секундная эпоха полисомнографической записи.

Публичный репозиторий: https://github.com/suh4rev/ucddb-apnea

## Что Реализовано

- загрузка и проверка данных UCDDB;
- построение 30-секундных эпох и признаков Flow, SpO2, усилий дыхания и ECG;
- subject-level cross-validation с группировкой по `record_id`;
- обучение ML-моделей и финального temporal ensemble;
- анализ интерпретируемости, ошибок FP/FN и стадий сна;
- Docker-контейнер, DVC-pipeline, optional MLflow/W&B tracking;
- benchmark инференса и экспорт финальной модели.

## Структура

```text
data/raw/             сырые данные UCDDB, не хранятся в Git
data/processed/       подготовленные таблицы признаков
reports/              отчеты, таблицы, рисунки и модельный артефакт
scripts/pipeline/     основной воспроизводимый pipeline
scripts/analysis/     анализ результатов и ошибок
scripts/experiments/  дополнительные эксперименты
scripts/mlops/        tracking, benchmark, экспорт модели
docs/                 карта pipeline и описание тюнинга
```

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Дополнительные зависимости:

```powershell
pip install -r requirements-dl.txt      # эксперименты с 1D-CNN/ResNet1D
pip install -r requirements-mlops.txt   # DVC, MLflow, W&B, Optuna, benchmark
```

## Основной Pipeline

Запускать из корня репозитория:

```powershell
python scripts/pipeline/00_download_ucddb.py
python scripts/pipeline/01_data_audit.py
python scripts/pipeline/02_build_epochs.py
python scripts/pipeline/03_extract_features.py
python scripts/pipeline/03_build_model_ready_features.py
python scripts/pipeline/03_validate_dataset.py
python scripts/pipeline/04_train_improved_models.py
python scripts/analysis/05_analyze_results.py
python scripts/pipeline/04_train_temporal_ensemble.py
python scripts/mlops/benchmark_inference.py
python scripts/analysis/06_error_analysis.py
python scripts/pipeline/06_audit_pipeline.py
python scripts/pipeline/06_interpret_final_models.py
python scripts/pipeline/07_make_final_artifacts.py
```

Короткая проверка готового проекта:

```powershell
python scripts/pipeline/03_validate_dataset.py
python scripts/pipeline/06_audit_pipeline.py
```

Последняя проверка дала:

```text
Validation: PASS 36, WARNING 0, FAIL 0
Pipeline audit: PASS 26, WARNING 0, FAIL 0
```

## DVC

Pipeline описан в `dvc.yaml`, зафиксированное состояние - в `dvc.lock`.

```powershell
dvc status
dvc dag
dvc repro --dry
```

Полезные стадии:

```powershell
dvc repro validate_dataset
dvc repro audit_pipeline
dvc repro benchmark_inference
dvc repro analyze_errors
dvc repro make_final_artifacts
```

Сырые медицинские данные не коммитятся. DVC используется для описания зависимостей, контроля стадий и воспроизводимости. Git LFS не является обязательным для текущей версии; при появлении крупных бинарных моделей их можно вынести в DVC remote или Git LFS.

## Docker

Сборка CPU-образа:

```powershell
docker build -t ucddb-apnea:core .
docker run --rm ucddb-apnea:core
```

Запуск pipeline-команды с локальными `data/` и `reports/`:

```powershell
docker run --rm `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\reports:/app/reports" `
  ucddb-apnea:core `
  python scripts/pipeline/03_validate_dataset.py
```

Опциональные варианты:

```powershell
docker build --build-arg REQUIREMENTS=requirements-dl.txt -t ucddb-apnea:dl .
docker build --build-arg REQUIREMENTS=requirements-mlops.txt -t ucddb-apnea:mlops .
```

## Tracking Экспериментов

MLflow и W&B выключены по умолчанию. Их можно включить через переменные окружения.

MLflow:

```powershell
$env:UCDDB_ENABLE_MLFLOW = "1"
python scripts/pipeline/04_train_temporal_ensemble.py
mlflow ui --backend-store-uri mlruns
```

W&B offline:

```powershell
$env:UCDDB_ENABLE_WANDB = "1"
$env:WANDB_MODE = "offline"
python scripts/pipeline/04_train_temporal_ensemble.py
```

Логируются параметры запуска, схема валидации, размеры данных, метрики ROC-AUC/F1, threshold tuning и итоговые CSV/Markdown артефакты.

## Подбор Гиперпараметров

Краткое описание тюнинга: [docs/hyperparameter_tuning.md](docs/hyperparameter_tuning.md).

Фактически использованы ограниченные grid/manual searches с grouped CV по пациентам:

- 5-fold subject-level CV по `record_id`;
- выбор по ROC-AUC, дополнительный анализ F1, sensitivity, specificity, AP;
- XGBoost grid по `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `reg_lambda`, `subsample`, `colsample_bytree`;
- post-processing search: окна сглаживания 15, 31, 61 эпох;
- threshold grid: 0.05-0.95 с шагом 0.05.

Optuna добавлена в MLOps-зависимости как опциональное расширение, но финальный результат получен контролируемым поиском конфигураций.

## Результаты

Финальная модель: temporal ensemble с centered rolling smoothing.

```text
ROC-AUC:      0.7066
F1 @ 0.50:    0.4349
F1 @ 0.45:    0.4502
Sensitivity:  0.6556
Specificity:  0.6516
```

Centered smoothing использует будущие вероятности внутри той же PSG-записи, поэтому этот вариант нужно описывать как offline retrospective analysis. Для сценария без будущей информации использовать causal smoothing.

## Инженерные Метрики

Benchmark финального ensemble:

```text
Sleep-only subset:        16067 эпох, 25 записей
Full inference time:      0.205563 s
Latency per epoch:        0.012794 ms
Batch latency per record: 8.222519 ms
Single-record latency:    67.764430 ms
Peak Python allocation:   19.153 MB
Model artifact size:      0.658 MB
GPU required:             no
```

Отчет: [reports/performance_report.md](reports/performance_report.md)

## Анализ Ошибок

Для финального threshold `0.45`:

```text
TP: 2286
FP: 4383
FN: 1201
TN: 8197
```

Основные наблюдения:

- FP чаще связаны с десатурационно-похожими паттернами: mean SpO2 min 89.60 против 91.61 у TN;
- FN чаще имеют менее выраженное падение SpO2: mean SpO2 min 91.63 против 88.80 у TP;
- ошибки концентрируются на отдельных записях, что указывает на влияние индивидуальной физиологии и качества сигнала;
- распределение ошибок отличается по стадиям сна.

Отчет: [reports/error_analysis_report.md](reports/error_analysis_report.md)

## Ключевые Артефакты

```text
reports/final_artifact_index.md
reports/tables/final_master_results_table.csv
reports/tables/final_practical_process_metrics.csv
reports/performance_report.md
reports/error_analysis_report.md
reports/tables/inference_benchmark.csv
reports/tables/error_analysis_by_record.csv
reports/tables/error_analysis_by_sleep_stage.csv
reports/models/final_temporal_ensemble_components.joblib
```

## Данные

Используется PhysioNet UCDDB. Скрипт загрузки сохраняет только необходимые файлы:

```text
RECORDS
SubjectDetails.xls
SHA256SUMS.txt
ucddb*.rec
ucddb*_respevt.txt
ucddb*_stage.txt
```

Большие `*_lifecard.edf` файлы не загружаются. Сырые данные находятся в `data/raw/` и не хранятся в Git.
