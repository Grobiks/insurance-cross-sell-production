"""Loading, validating, cleaning and splitting the dataset."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when input data does not match the expected schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


HTTP_TIMEOUT = 60  # seconds per request
RANGE_CHUNK = 4 * (1 << 20)


def _download_stream(url: str, out_path: Path) -> None:
    """Single connection; used when the file size is unknown."""
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response, open(out_path, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)


def _download_ranges(url: str, out_path: Path, size: int, workers: int) -> None:
    """Fetch the file as parallel HTTP Range requests.

    Some networks throttle each connection to a few dozen KB/s (observed with GitHub
    LFS), while many parallel connections reach the full bandwidth.
    """
    with open(out_path, "wb") as f:
        f.truncate(size)
    lock = threading.Lock()
    done = 0

    def fetch(start: int) -> None:
        nonlocal done
        end = min(start + RANGE_CHUNK, size) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError(f"expected {end - start + 1} bytes, got {len(data)}")
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        with lock:
            with open(out_path, "r+b") as f:
                f.seek(start)
                f.write(data)
            done += len(data)
            logger.info("  %d / %d MB", done >> 20, size >> 20)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, range(0, size, RANGE_CHUNK)))


def download_data(
    path: Path = config.DATA_PATH,
    url: str = config.DATA_URL,
    force: bool = False,
    sha256: str | None = config.DATA_SHA256,
    size: int | None = config.DATA_SIZE_BYTES,
    workers: int = 32,
) -> Path:
    """Download the raw CSV once, verify its checksum and cache it on disk.

    With a known `size` and `workers > 1` the file is fetched in parallel ranges.
    """
    path = Path(path)
    if path.exists() and not force:
        logger.info("Data already present at %s, skipping download", path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading %s -> %s (%d workers)", url, path, workers)
    if size and workers > 1:
        _download_ranges(url, tmp_path, size, workers)
    else:
        _download_stream(url, tmp_path)

    if sha256 is not None and _sha256(tmp_path) != sha256:
        tmp_path.unlink()
        raise OSError(f"Checksum mismatch for {url}: the downloaded file is corrupted")
    tmp_path.replace(path)
    return path


CSV_DTYPES = {
    "Age": "int16",
    "Driving_License": "int8",
    "Previously_Insured": "int8",
    "Vintage": "int16",
    "Response": "int8",
    "Annual_Premium": "float32",
    # Region_Code stays float64: 39.2 must compare exactly equal to the anomaly marker
}
CHUNK_ROWS = 1_000_000


def load_sample(
    path: Path = config.DATA_PATH,
    frac: float = config.SAMPLE_FRAC,
    random_state: int = config.RANDOM_STATE,
    nrows: int | None = None,
    chunksize: int = CHUNK_ROWS,
) -> pd.DataFrame:
    """Read the CSV in chunks, clean and stratified-subsample each chunk.

    Peak memory is one chunk plus the sample, instead of the whole 11.5M-row frame.
    Duplicates are only removed within a chunk (the EDA found none in the full data).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `insurance download` or pass --data-path.")
    reader = pd.read_csv(
        path, index_col=config.ID_COLUMN, dtype=CSV_DTYPES, nrows=nrows, chunksize=chunksize
    )
    parts = [stratified_sample(clean(chunk), frac, random_state) for chunk in reader]
    return pd.concat(parts)


def validate_features(df: pd.DataFrame) -> None:
    """Check that `df` has every feature column and only allowed categorical values."""
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    with_nans = [c for c in config.FEATURE_COLUMNS if df[c].isna().any()]
    if with_nans:
        raise SchemaError(f"Missing values in columns: {with_nans}")

    for column, allowed in config.ALLOWED_VALUES.items():
        unexpected = set(df[column].dropna().unique()) - allowed
        if unexpected:
            raise SchemaError(
                f"Column {column!r} has unexpected values: {sorted(unexpected, key=str)}"
            )

    numeric = [*config.OUTLIER_NUM_COLS, *config.NUM_COLS, *config.HIGH_CARDINALITY_COLS]
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise SchemaError(f"Column {column!r} must be numeric, got {df[column].dtype}")
        if (df[column] < 0).any():
            raise SchemaError(f"Column {column!r} contains negative values")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Training-time cleaning: drop duplicates and the single anomalous region row.

    The EDA found no missing values; if new data introduces NaNs the rows are
    dropped here rather than silently passed to the models.
    """
    before = len(df)
    df = df[df["Region_Code"] != config.ANOMALOUS_REGION_CODE]
    df = df.dropna(subset=[*config.FEATURE_COLUMNS, config.TARGET])
    df = df.drop_duplicates()
    logger.info("Cleaning removed %d rows (%d -> %d)", before - len(df), before, len(df))
    return df


def stratified_sample(
    df: pd.DataFrame, frac: float, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Subsample while preserving the class balance of the target."""
    if frac >= 1.0:
        return df
    parts = [
        group.sample(frac=frac, random_state=random_state)
        for _, group in df.groupby(config.TARGET, sort=False)
    ]
    return pd.concat(parts)


def split_xy(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split; split happens before any fitting (no leakage)."""
    validate_features(df)
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
