from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it before running the ResNet1D experiment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_TABLES_DIR  # noqa: E402


RANDOM_STATE = 42
N_SPLITS = 5
TARGET_FS = 8.0
THRESHOLD = 0.5

BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PREVIOUS_CNN_AUC = 0.5953
PREVIOUS_CNN_F1 = 0.3453
TEMPORAL_ENSEMBLE_AUC = 0.7066
TEMPORAL_ENSEMBLE_F1 = 0.4349

EPOCHS_PATH = DATA_PROCESSED_DIR / "epochs.csv"
BASELINE_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "improved_cv_predictions.csv"

RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "cnn_improved_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "cnn_improved_results_summary.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "cnn_improved_best_thresholds.csv"
CV_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "cnn_improved_cv_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "cnn_improved_training_report.md"

CHANNELS = ("flow", "spo2", "ribcage", "abdo")
CHANNEL_ALIASES = {
    "flow": {"flow"},
    "spo2": {"spo2", "sp02"},
    "ribcage": {"ribcage"},
    "abdo": {"abdo"},
}

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "average_precision",
]
CONFUSION_COLUMNS = ["tn", "fp", "fn", "tp"]
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)


@dataclass(frozen=True)
class InputConfig:
    input_mode: str
    window_sec: int
    start_offset_sec: int
    offline_retrospective: bool

    @property
    def target_length(self) -> int:
        return int(self.window_sec * TARGET_FS)

    @property
    def input_shape(self) -> str:
        return f"{len(CHANNELS)}x{self.target_length}"


INPUT_CONFIG = InputConfig(
    input_mode="cnn_150s_context",
    window_sec=150,
    start_offset_sec=-60,
    offline_retrospective=True,
)

POSTPROCESSING_CONFIGS = [
    {
        "postprocessing": "raw",
        "smoothing_window_epochs": 0,
        "smoothing_centered": False,
        "offline_retrospective_smoothing": False,
    },
    {
        "postprocessing": "rolling_mean_causal",
        "smoothing_window_epochs": 15,
        "smoothing_centered": False,
        "offline_retrospective_smoothing": False,
    },
    {
        "postprocessing": "rolling_mean_centered",
        "smoothing_window_epochs": 15,
        "smoothing_centered": True,
        "offline_retrospective_smoothing": True,
    },
    {
        "postprocessing": "rolling_mean_causal",
        "smoothing_window_epochs": 31,
        "smoothing_centered": False,
        "offline_retrospective_smoothing": False,
    },
    {
        "postprocessing": "rolling_mean_centered",
        "smoothing_window_epochs": 31,
        "smoothing_centered": True,
        "offline_retrospective_smoothing": True,
    },
]

GROUP_COLUMNS = [
    "input_mode",
    "input_shape",
    "postprocessing",
    "smoothing_window_epochs",
    "smoothing_centered",
    "offline_retrospective",
    "offline_retrospective_smoothing",
]


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_pool: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) if use_pool else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        out = self.pool(out)
        return self.dropout(out)


class ResNet1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.block1 = ResidualBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=7,
            dropout=0.1,
            use_pool=True,
        )
        self.block2 = ResidualBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            dropout=0.15,
            use_pool=True,
        )
        self.block3 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)


def import_pyedflib():
    try:
        import pyedflib
    except ImportError as exc:
        raise SystemExit(
            "pyedflib is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    return pyedflib


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize_channel_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def find_channel_indices(labels: list[str]) -> dict[str, int]:
    normalized_labels = {
        normalize_channel_name(label): index for index, label in enumerate(labels)
    }
    indices: dict[str, int] = {}

    for channel, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            if alias in normalized_labels:
                indices[channel] = normalized_labels[alias]
                break

    return indices


def find_record_file(record_id: str) -> Path:
    direct_path = DATA_RAW_DIR / f"{record_id}.rec"
    if direct_path.exists():
        return direct_path

    matches = sorted(
        path
        for path in DATA_RAW_DIR.glob(f"{record_id}*.rec")
        if path.is_file() and "lifecard" not in path.name.lower()
    )
    if not matches:
        raise SystemExit(f"Missing .rec file for record {record_id}")

    return matches[0]


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    median = float(np.nanmedian(finite))
    q25, q75 = np.nanpercentile(finite, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.nanstd(finite))
    if not np.isfinite(scale) or scale == 0:
        return np.nan_to_num(values, nan=0.0).astype(np.float32)

    normalized = (values - median) / scale
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def prepare_spo2(values: np.ndarray) -> np.ndarray:
    spo2 = np.asarray(values, dtype=np.float32).copy()
    spo2[(spo2 < 50) | (spo2 > 100)] = np.nan

    series = pd.Series(spo2)
    filled = series.interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    if np.isnan(filled).all():
        filled = np.full_like(filled, 90.0, dtype=np.float32)
    elif np.isnan(filled).any():
        median = float(np.nanmedian(filled))
        filled = np.nan_to_num(filled, nan=median).astype(np.float32)

    normalized = (filled - 90.0) / 10.0
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def read_preprocessed_record(record_id: str) -> dict[str, dict[str, np.ndarray | float]]:
    pyedflib = import_pyedflib()
    record_path = find_record_file(record_id)
    signals: dict[str, dict[str, np.ndarray | float]] = {}

    with pyedflib.EdfReader(str(record_path)) as reader:
        labels = reader.getSignalLabels()
        frequencies = reader.getSampleFrequencies()
        channel_indices = find_channel_indices(labels)
        missing_channels = sorted(set(CHANNELS) - set(channel_indices))
        if missing_channels:
            raise SystemExit(
                f"{record_id}: missing required channels {missing_channels}"
            )

        for channel in CHANNELS:
            channel_index = channel_indices[channel]
            raw = np.asarray(reader.readSignal(channel_index), dtype=np.float32)
            fs = float(frequencies[channel_index])
            if channel == "spo2":
                prepared = prepare_spo2(raw)
            else:
                prepared = robust_normalize(raw)

            signals[channel] = {
                "signal": prepared,
                "time": np.arange(prepared.size, dtype=np.float32) / fs,
                "fs": fs,
            }

    return signals


def extract_window(
    channel_data: dict[str, np.ndarray | float],
    start_sec: float,
    target_length: int,
) -> np.ndarray:
    signal = channel_data["signal"]
    time_axis = channel_data["time"]
    if not isinstance(signal, np.ndarray) or not isinstance(time_axis, np.ndarray):
        raise TypeError("channel_data must contain numpy signal and time arrays")

    target_times = start_sec + np.arange(target_length, dtype=np.float32) / TARGET_FS
    segment = np.interp(target_times, time_axis, signal, left=0.0, right=0.0)
    return segment.astype(np.float32)


def load_sleep_epochs() -> pd.DataFrame:
    if not EPOCHS_PATH.exists():
        raise SystemExit(
            f"Epoch table not found: {EPOCHS_PATH}. "
            "Run: python scripts/pipeline/02_build_epochs.py"
        )

    epochs = pd.read_csv(EPOCHS_PATH)
    required = {
        "record_id",
        "epoch_id",
        "start_sec",
        "end_sec",
        "sleep_stage",
        "label_binary",
    }
    missing = required - set(epochs.columns)
    if missing:
        raise SystemExit(f"Missing required columns in epochs.csv: {sorted(missing)}")

    sleep_epochs = epochs[
        epochs["sleep_stage"].notna()
        & (epochs["sleep_stage"] != 0)
        & (epochs["sleep_stage"] != 8)
    ].copy()
    sleep_epochs["record_id"] = sleep_epochs["record_id"].astype(str).str.lower()
    sleep_epochs["label_binary"] = sleep_epochs["label_binary"].astype(int)
    sleep_epochs = sleep_epochs.sort_values(
        ["record_id", "start_sec", "epoch_id"]
    ).reset_index(drop=True)

    labels = set(sleep_epochs["label_binary"].unique())
    if not labels.issubset({0, 1}):
        raise SystemExit(f"label_binary must contain only 0/1, got {sorted(labels)}")

    return sleep_epochs


def load_official_fold_map(sleep_epochs: pd.DataFrame) -> dict[str, int] | None:
    if not BASELINE_PREDICTIONS_PATH.exists():
        return None

    predictions = pd.read_csv(
        BASELINE_PREDICTIONS_PATH,
        usecols=["record_id", "regime", "feature_variant", "experiment", "fold"],
    )
    predictions["record_id"] = predictions["record_id"].astype(str).str.lower()
    baseline = predictions[
        (predictions["regime"] == "sleep_only")
        & (predictions["feature_variant"] == "enhanced")
        & (predictions["experiment"] == "spo2_only")
    ].copy()
    if baseline.empty:
        return None

    fold_counts = baseline.groupby("record_id")["fold"].nunique()
    if (fold_counts > 1).any():
        return None

    fold_map = (
        baseline[["record_id", "fold"]]
        .drop_duplicates()
        .set_index("record_id")["fold"]
        .astype(int)
        .to_dict()
    )
    records = set(sleep_epochs["record_id"].unique())
    if not records.issubset(fold_map):
        return None

    return fold_map


def make_fallback_fold_map(sleep_epochs: pd.DataFrame) -> dict[str, int]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        splitter = GroupKFold(n_splits=N_SPLITS)

    y = sleep_epochs["label_binary"].astype(int)
    groups = sleep_epochs["record_id"].astype(str)
    dummy_x = np.zeros((len(sleep_epochs), 1), dtype=np.float32)
    fold_map: dict[str, int] = {}

    for fold, (_, valid_indices) in enumerate(
        splitter.split(dummy_x, y, groups), start=1
    ):
        valid_records = groups.iloc[valid_indices].unique()
        for record_id in valid_records:
            fold_map[str(record_id)] = fold

    return fold_map


def make_fold_map(sleep_epochs: pd.DataFrame) -> tuple[dict[str, int], str]:
    official = load_official_fold_map(sleep_epochs)
    if official is not None:
        return official, "saved improved baseline folds"

    return make_fallback_fold_map(sleep_epochs), "fresh subject-level CV folds"


def make_fold_indices(
    sleep_epochs: pd.DataFrame,
    fold_map: dict[str, int],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    folds = sleep_epochs["record_id"].map(fold_map)
    if folds.isna().any():
        missing = sorted(sleep_epochs.loc[folds.isna(), "record_id"].unique())
        raise SystemExit(f"Missing fold assignment for records: {missing}")

    fold_values = folds.astype(int).to_numpy()
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold in sorted(np.unique(fold_values)):
        valid_indices = np.flatnonzero(fold_values == fold)
        train_indices = np.flatnonzero(fold_values != fold)
        if len(valid_indices) == 0 or len(train_indices) == 0:
            raise SystemExit(f"Invalid empty train/validation split for fold {fold}")
        splits.append((int(fold), train_indices, valid_indices))

    return splits


def build_input_tensor(sleep_epochs: pd.DataFrame) -> np.ndarray:
    n_rows = len(sleep_epochs)
    x = np.zeros(
        (n_rows, len(CHANNELS), INPUT_CONFIG.target_length),
        dtype=np.float32,
    )

    grouped_indices = sleep_epochs.groupby("record_id", sort=True).indices
    iterator = tqdm(
        grouped_indices.items(),
        total=len(grouped_indices),
        desc=f"Building {INPUT_CONFIG.input_mode}",
    )
    for record_id, row_indices in iterator:
        record_signals = read_preprocessed_record(str(record_id))
        rows = sleep_epochs.iloc[row_indices]
        for position, row in zip(row_indices, rows.itertuples(index=False)):
            start_sec = float(row.start_sec) + INPUT_CONFIG.start_offset_sec
            for channel_index, channel in enumerate(CHANNELS):
                x[position, channel_index] = extract_window(
                    record_signals[channel],
                    start_sec=start_sec,
                    target_length=INPUT_CONFIG.target_length,
                )

    return x


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return np.nan


def safe_average_precision(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        return np.nan


def calculate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float | int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_from_confusion(int(tn), int(fp))),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "average_precision": safe_average_precision(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def calculate_pos_weight(y_train: np.ndarray) -> float:
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            if criterion is not None:
                loss = criterion(logits, batch_y)
                losses.append(float(loss.item()) * int(batch_y.numel()))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_y.cpu().numpy())

    y_true = np.concatenate(labels).astype(int)
    y_proba = np.concatenate(probabilities).astype(np.float32)
    val_loss = float(np.sum(losses) / len(y_true)) if losses else np.nan

    return y_true, y_proba, val_loss


def train_one_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, int]:
    set_random_seed(seed)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y.astype(np.float32)),
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ResNet1D().to(device)
    pos_weight = calculate_pos_weight(y[train_indices])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_proba: np.ndarray | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        y_valid, y_proba, val_loss = predict_proba(
            model,
            valid_loader,
            device,
            criterion,
        )
        val_auc = safe_roc_auc(y_valid, y_proba)
        score = val_auc if np.isfinite(val_auc) else -val_loss

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_proba = y_proba.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        _, best_proba, _ = predict_proba(model, valid_loader, device, criterion)

    if best_proba is None:
        raise RuntimeError("ResNet1D training did not produce validation predictions")

    return best_proba, best_epoch


def smooth_probabilities(
    raw_predictions: pd.DataFrame,
    window: int,
    centered: bool,
) -> pd.DataFrame:
    smoothed = raw_predictions.copy()
    smoothed = smoothed.sort_values(["record_id", "start_sec", "epoch_id"]).copy()
    smoothed["y_proba"] = (
        smoothed.groupby("record_id", group_keys=False)["y_proba"]
        .apply(
            lambda series: series.rolling(
                window=window,
                min_periods=1,
                center=centered,
            ).mean()
        )
        .to_numpy()
    )
    smoothed["y_pred"] = (smoothed["y_proba"] >= THRESHOLD).astype(int)
    return smoothed


def make_postprocessed_predictions(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config in POSTPROCESSING_CONFIGS:
        if config["postprocessing"] == "raw":
            current = raw_predictions.copy()
        else:
            current = smooth_probabilities(
                raw_predictions,
                window=int(config["smoothing_window_epochs"]),
                centered=bool(config["smoothing_centered"]),
            )
        current["postprocessing"] = config["postprocessing"]
        current["smoothing_window_epochs"] = config["smoothing_window_epochs"]
        current["smoothing_centered"] = config["smoothing_centered"]
        current["offline_retrospective"] = INPUT_CONFIG.offline_retrospective
        current["offline_retrospective_smoothing"] = config[
            "offline_retrospective_smoothing"
        ]
        frames.append(current)

    return pd.concat(frames, ignore_index=True)


def calculate_results_by_fold(
    predictions: pd.DataFrame,
    best_epochs_by_fold: dict[int, int],
    fold_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in predictions.groupby([*GROUP_COLUMNS, "fold"], sort=False):
        group_dict = dict(zip([*GROUP_COLUMNS, "fold"], group_values))
        y_true = group["label_binary"].to_numpy(dtype=int)
        y_proba = group["y_proba"].to_numpy(dtype=float)
        metrics = calculate_metrics(y_true, y_proba, THRESHOLD)
        train_records = group["n_train_records"].iloc[0]
        rows.append(
            {
                **group_dict,
                "fold_source": fold_source,
                "n_train": int(group["n_train"].iloc[0]),
                "n_valid": int(len(group)),
                "n_train_records": int(train_records),
                "n_valid_records": int(group["record_id"].nunique()),
                "best_epoch": int(best_epochs_by_fold[int(group_dict["fold"])]),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_results(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for group_values, group in results_by_fold.groupby(GROUP_COLUMNS, sort=False):
        group_dict = dict(zip(GROUP_COLUMNS, group_values))
        row: dict[str, object] = {
            **group_dict,
            "fold_source": group["fold_source"].iloc[0],
            "n_folds": int(group["fold"].nunique()),
            "n_epochs": int(group["n_valid"].sum()),
            "n_records": int(group["n_valid_records"].sum()),
            "best_epoch_mean": float(group["best_epoch"].mean()),
            "best_epoch_std": float(group["best_epoch"].std(ddof=0)),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        for column in CONFUSION_COLUMNS:
            row[f"{column}_sum"] = int(group[column].sum())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("roc_auc_mean", ascending=False)


def sweep_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in predictions.groupby(GROUP_COLUMNS, sort=False):
        group_dict = dict(zip(GROUP_COLUMNS, group_values))
        y_true = group["label_binary"].to_numpy(dtype=int)
        y_proba = group["y_proba"].to_numpy(dtype=float)
        sweep_rows = []
        for threshold in THRESHOLDS:
            metrics = calculate_metrics(y_true, y_proba, float(threshold))
            sweep_rows.append(
                {
                    "threshold": float(threshold),
                    "youden_index": (
                        metrics["sensitivity"] + metrics["specificity"] - 1
                    ),
                    **metrics,
                }
            )

        sweep = pd.DataFrame(sweep_rows)
        selections = [
            ("max_f1", sweep.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]),
            (
                "max_youden",
                sweep.sort_values(["youden_index", "f1"], ascending=False).iloc[0],
            ),
        ]
        sensitivity_candidates = sweep[sweep["sensitivity"] >= 0.70]
        if sensitivity_candidates.empty:
            row = sweep.sort_values(
                ["sensitivity", "specificity"],
                ascending=False,
            ).iloc[0]
            rule = "sensitivity_ge_0_70_max_specificity_unmet"
        else:
            row = sensitivity_candidates.sort_values(
                ["specificity", "f1"],
                ascending=False,
            ).iloc[0]
            rule = "sensitivity_ge_0_70_max_specificity"
        selections.append((rule, row))

        for selection_rule, row in selections:
            rows.append(
                {
                    **group_dict,
                    "selection_rule": selection_rule,
                    "threshold": float(row["threshold"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "sensitivity": float(row["sensitivity"]),
                    "specificity": float(row["specificity"]),
                    "f1": float(row["f1"]),
                    "tn": int(row["tn"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tp": int(row["tp"]),
                    "youden_index": float(row["youden_index"]),
                }
            )

    return pd.DataFrame(rows)


def dataframe_to_markdown(table: pd.DataFrame, decimals: int = 4) -> str:
    if table.empty:
        return "No rows available."

    rounded = table.copy()
    numeric_columns = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(decimals)

    headers = list(rounded.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in rounded.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_report(
    results_summary: pd.DataFrame,
    best_thresholds: pd.DataFrame,
    fold_source: str,
    device: torch.device,
) -> None:
    best_auc_row = results_summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_f1_row = results_summary.sort_values("f1_mean", ascending=False).iloc[0]
    max_f1_rows = best_thresholds[best_thresholds["selection_rule"] == "max_f1"]
    best_tuned_row = max_f1_rows.sort_values("f1", ascending=False).iloc[0]

    best_auc = float(best_auc_row["roc_auc_mean"])
    best_f1 = float(best_f1_row["f1_mean"])
    best_tuned_f1 = float(best_tuned_row["f1"])

    if best_auc > TEMPORAL_ENSEMBLE_AUC:
        comparison_text = (
            "ResNet1D improved over the temporal ensemble on internal UCDDB CV. "
            "Because this is a small single-dataset result, it needs external "
            "validation before any clinical interpretation."
        )
    elif best_auc > PREVIOUS_CNN_AUC:
        comparison_text = (
            "ResNet1D improved over the simpler 1D-CNN baseline, but did not beat "
            "the temporal ensemble. This is plausible with only 25 subjects, "
            "strict subject-level validation, and no large-scale pretraining."
        )
    else:
        comparison_text = (
            "ResNet1D did not improve over the simpler 1D-CNN baseline or the "
            "temporal ensemble. This is plausible with only 25 subjects, strict "
            "subject-level validation, and no large-scale pretraining."
        )

    summary_table = dataframe_to_markdown(
        results_summary[
            [
                "postprocessing",
                "smoothing_window_epochs",
                "smoothing_centered",
                "roc_auc_mean",
                "f1_mean",
                "sensitivity_mean",
                "specificity_mean",
                "average_precision_mean",
            ]
        ],
        decimals=4,
    )
    threshold_table = dataframe_to_markdown(
        best_thresholds[
            [
                "postprocessing",
                "smoothing_window_epochs",
                "selection_rule",
                "threshold",
                "f1",
                "sensitivity",
                "specificity",
            ]
        ],
        decimals=4,
    )

    report = f"""# Improved 1D-CNN Training Report

## Goal

Controlled DL improvement experiment for UCDDB sleep-only binary classification: test whether longer offline context plus a residual CNN improves the simpler raw-signal 1D-CNN baseline.

## Data And Leakage Controls

- Input files: `data/processed/epochs.csv` and `data/raw/ucddb*.rec`.
- Signals: Flow, SpO2, ribcage, abdo.
- Excluded from model input: ECG, sleep stage, record/epoch/time identifiers, labels, and event metadata.
- Sleep stage is used only to select sleep-only epochs.
- Cross-validation: subject-level 5-fold CV by `record_id`.
- Fold source: {fold_source}.

## Input Mode

- `cnn_150s_context`: prev2 + prev1 + current + next1 + next2 epochs.
- Input shape: `4 x 1200`.
- Label belongs to the central epoch.
- This is an offline retrospective mode because it uses future neighboring epochs. Centered smoothing is also offline retrospective because it uses future probabilities within a record.

## Architecture

Stem: `Conv1d(4, 32, kernel_size=7, padding=3)` -> BatchNorm -> ReLU.

Residual block 1: two `Conv1d(32, 32, kernel_size=7, padding=3)` layers, residual connection, ReLU, MaxPool, Dropout(0.1).

Residual block 2: two `Conv1d(..., 64, kernel_size=5, padding=2)` layers, residual projection 32->64, ReLU, MaxPool, Dropout(0.15).

Residual block 3: two `Conv1d(..., 128, kernel_size=3, padding=1)` layers, residual projection 64->128, ReLU.

Head: AdaptiveAvgPool1d(1) -> Linear(128, 1).

Training used BCEWithLogitsLoss with fold-specific `pos_weight`, Adam (`lr=1e-3`, `weight_decay=1e-4`), batch size 128, maximum 40 epochs, and early stopping patience 6 by validation ROC-AUC. WeightedRandomSampler was not used.

Device used: `{device}`.

## Results At Threshold 0.5

{summary_table}

## Threshold Sweep

{threshold_table}

## Comparison

Previous simple CNN: ROC-AUC={PREVIOUS_CNN_AUC:.4f}, F1={PREVIOUS_CNN_F1:.4f}.

Temporal ensemble: ROC-AUC={TEMPORAL_ENSEMBLE_AUC:.4f}, F1={TEMPORAL_ENSEMBLE_F1:.4f}.

Best ResNet1D ROC-AUC: {best_auc:.4f}.

Best ResNet1D F1 at threshold 0.5: {best_f1:.4f}.

Best tuned ResNet1D F1: {best_tuned_f1:.4f}.

{comparison_text}

## Limitations

- UCDDB is small: only 25 subjects.
- Subject-level CV is intentionally strict and can produce high fold variance.
- The model is a controlled residual CNN baseline, not a large DL search.
- No external validation is performed here.
- `cnn_150s_context` and centered smoothing are offline retrospective and are not suitable for real-time use.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    set_random_seed(RANDOM_STATE)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sleep_epochs = load_sleep_epochs()
    fold_map, fold_source = make_fold_map(sleep_epochs)
    folds = make_fold_indices(sleep_epochs, fold_map)

    y = sleep_epochs["label_binary"].to_numpy(dtype=int)
    records = sleep_epochs["record_id"].astype(str).to_numpy()
    epoch_ids = sleep_epochs["epoch_id"].to_numpy(dtype=int)
    start_secs = sleep_epochs["start_sec"].to_numpy(dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loaded {len(sleep_epochs)} sleep-only epochs from {len(set(records))} records.")
    print(f"Using {fold_source}.")
    print(f"Training device: {device}.")
    print(
        f"Input mode: {INPUT_CONFIG.input_mode}, shape {INPUT_CONFIG.input_shape}."
    )

    x = build_input_tensor(sleep_epochs)
    raw_prediction_frames: list[pd.DataFrame] = []
    best_epochs_by_fold: dict[int, int] = {}

    for fold, train_indices, valid_indices in folds:
        train_records = sorted(set(records[train_indices]))
        valid_records = sorted(set(records[valid_indices]))
        print(
            f"{INPUT_CONFIG.input_mode} fold {fold}: "
            f"{len(train_indices)} train epochs, {len(valid_indices)} valid epochs"
        )

        y_proba, best_epoch = train_one_fold(
            x=x,
            y=y,
            train_indices=train_indices,
            valid_indices=valid_indices,
            device=device,
            seed=RANDOM_STATE + int(fold),
        )
        best_epochs_by_fold[int(fold)] = int(best_epoch)

        y_valid = y[valid_indices]
        raw_prediction_frames.append(
            pd.DataFrame(
                {
                    "record_id": records[valid_indices],
                    "epoch_id": epoch_ids[valid_indices],
                    "start_sec": start_secs[valid_indices],
                    "label_binary": y_valid,
                    "input_mode": INPUT_CONFIG.input_mode,
                    "input_shape": INPUT_CONFIG.input_shape,
                    "fold": int(fold),
                    "n_train": int(len(train_indices)),
                    "n_train_records": int(len(train_records)),
                    "n_valid_records": int(len(valid_records)),
                    "y_proba": y_proba,
                    "y_pred": (y_proba >= THRESHOLD).astype(int),
                }
            )
        )

    raw_predictions = pd.concat(raw_prediction_frames, ignore_index=True)
    predictions = make_postprocessed_predictions(raw_predictions)
    results_by_fold = calculate_results_by_fold(
        predictions,
        best_epochs_by_fold=best_epochs_by_fold,
        fold_source=fold_source,
    )
    results_summary = summarize_results(results_by_fold)
    best_thresholds = sweep_thresholds(predictions)

    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    results_summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    best_thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)
    predictions.to_csv(CV_PREDICTIONS_PATH, index=False)
    write_report(results_summary, best_thresholds, fold_source, device)

    best_auc_row = results_summary.iloc[0]
    best_f1_row = results_summary.sort_values("f1_mean", ascending=False).iloc[0]
    best_tuned = best_thresholds[
        best_thresholds["selection_rule"] == "max_f1"
    ].sort_values("f1", ascending=False).iloc[0]

    print("\nImproved ResNet1D experiment complete.")
    print(
        "Best ROC-AUC: "
        f"{best_auc_row['roc_auc_mean']:.4f} "
        f"({best_auc_row['postprocessing']}, "
        f"window={best_auc_row['smoothing_window_epochs']})"
    )
    print(
        "Best F1 @ 0.5: "
        f"{best_f1_row['f1_mean']:.4f} "
        f"({best_f1_row['postprocessing']}, "
        f"window={best_f1_row['smoothing_window_epochs']})"
    )
    print(
        "Best tuned F1: "
        f"{best_tuned['f1']:.4f} ({best_tuned['postprocessing']}, "
        f"window={best_tuned['smoothing_window_epochs']}, "
        f"threshold={best_tuned['threshold']:.2f})"
    )
    print(
        "Reference results: "
        f"previous CNN AUC={PREVIOUS_CNN_AUC:.4f}, F1={PREVIOUS_CNN_F1:.4f}; "
        f"temporal ensemble AUC={TEMPORAL_ENSEMBLE_AUC:.4f}, "
        f"F1={TEMPORAL_ENSEMBLE_F1:.4f}."
    )
    print(f"Saved: {RESULTS_SUMMARY_PATH}")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
