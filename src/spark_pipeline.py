"""PySpark implementation of the OASIS data pipeline.

Mirrors `src/data_pipeline.py` row-for-row so both can be run side by side —
see `scripts/verify_pipeline_parity.py`, which asserts they produce
identical patient splits and class distributions. File enumeration itself
stays a lightweight Python `os.listdir` walk (Spark's Hadoop-backed file
listing needs native bindings that aren't reliably available cross-platform,
and listing a few hundred thousand small image files isn't the part that
benefits from distribution anyway); everything downstream of that raw file
list — parsing patient/slice IDs, filtering, splitting, resampling — runs
as genuine Spark DataFrame operations (`regexp_extract`, `orderBy(rand())`,
`unionByName`, ...), which is the part a real cluster would actually help
with at OASIS's full 86K-row scale.
"""
import math
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from sklearn.model_selection import train_test_split
import numpy as np

from data_pipeline import CATEGORIES

# OASIS filename convention: OAS1_0001_MR1_mpr-1_150.jpg
_FILENAME_REGEX = r'OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+)\.jpg'


def get_spark(app_name="oasis-pipeline", master=None):
    """Local SparkSession, suitable for a single machine (Colab or laptop).

    On native Windows, PySpark's multi-threaded local executor
    (`local[*]`) is prone to flaky loopback-socket errors on session
    startup — a long-standing, widely-reported issue, likely from
    antivirus/firewall interference with local socket connections. When
    `master` isn't given, we default to `local[*]` everywhere except
    Windows, where we fall back to single-threaded `local[1]` and pin
    `PYSPARK_PYTHON`/`SPARK_LOCAL_IP`, which together avoid it reliably.
    """
    if master is None:
        master = "local[1]" if sys.platform == "win32" else "local[*]"

    if sys.platform == "win32":
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def create_metadata_df_spark(spark, base_path):
    """Build a Spark DataFrame of OASIS scan metadata: one row per MRI slice.

    Enumerates `base_path/<category>/*.jpg` with plain Python (same walk as
    `data_pipeline.create_metadata_df`), then hands the raw (path, filename,
    label, category) tuples to Spark, where `regexp_extract` pulls out
    patient_id/slice_id — matching `data_pipeline.create_metadata_df`'s
    columns exactly.
    """
    base_path = str(base_path)
    rows = []
    for folder, label in CATEGORIES.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            rows.append((os.path.join(folder_path, filename), filename, label, folder))

    raw_df = spark.createDataFrame(rows, schema=["path", "filename", "label", "category"])

    return (
        raw_df
        .withColumn("patient_id", F.regexp_extract("filename", _FILENAME_REGEX, 1))
        .withColumn("slice_id", F.regexp_extract("filename", _FILENAME_REGEX, 4).cast(IntegerType()))
        .filter(F.col("patient_id") != "")
        .select("path", "patient_id", "slice_id", "label", "category")
    )


def patient_level_split_spark(df, test_size=0.20, val_size=0.10, random_state=42):
    """Spark counterpart to `data_pipeline.patient_level_split`.

    Per-category patient lists are small enough to collect to the driver
    (at most a few hundred IDs), where the exact same sorted-list +
    `train_test_split`/`RandomState` decision logic as the pandas version
    runs. The result (which patient IDs go where) is then broadcast back
    to filter the full distributed DataFrame via `.isin(...)`. Splitting
    decisions need to run somewhere sequential regardless of backend; what
    Spark actually parallelizes is the large per-row filter/join over the
    full 86K-row metadata table, not the patient-selection logic itself.
    """
    train_pats, val_pats, test_pats = [], [], []

    for category in CATEGORIES:
        cat_patients = sorted(
            row["patient_id"]
            for row in df.filter(F.col("category") == category).select("patient_id").distinct().collect()
        )
        n = len(cat_patients)

        if n <= 2:
            train_pats.extend(cat_patients)
            val_pats.extend(cat_patients)
            test_pats.extend(cat_patients)
        elif n < 6:
            shuffled = np.random.RandomState(random_state).permutation(cat_patients)
            test_pats.extend(shuffled[:1])
            val_pats.extend(shuffled[1:2])
            train_pats.extend(shuffled[2:])
        else:
            tr, te = train_test_split(cat_patients, test_size=test_size, random_state=random_state)
            tr, va = train_test_split(tr, test_size=val_size, random_state=random_state)
            train_pats.extend(tr)
            val_pats.extend(va)
            test_pats.extend(te)

    train_df = df.filter(F.col("patient_id").isin(train_pats))
    val_df = df.filter(F.col("patient_id").isin(val_pats))
    test_df = df.filter(F.col("patient_id").isin(test_pats))
    return train_df, val_df, test_df


def hybrid_resample_spark(df, target_samples_per_class=8000, random_state=42):
    """Spark counterpart to `data_pipeline.hybrid_resample`.

    Under-sampling is an exact `orderBy(rand(seed)).limit(target)`.
    Over-sampling replicates the category's rows enough times to cover the
    target count, then applies the same `orderBy(rand(seed)).limit(target)`
    to land on an exact count with duplicated rows — analogous to pandas'
    `sample(..., replace=True)`, though the specific duplicate rows chosen
    won't match 1:1 with the pandas version since Spark's and numpy's RNGs
    are different implementations. Parity is checked at the class-count
    level, not the row level, for this step.
    """
    segments = []
    for category in CATEGORIES:
        subset = df.filter(F.col("category") == category)
        n = subset.count()

        if n >= target_samples_per_class:
            sampled = subset.orderBy(F.rand(seed=random_state)).limit(target_samples_per_class)
        else:
            reps = math.ceil(target_samples_per_class / n)
            replicated = subset
            for _ in range(reps - 1):
                replicated = replicated.unionByName(subset)
            sampled = replicated.orderBy(F.rand(seed=random_state)).limit(target_samples_per_class)

        segments.append(sampled)

    result = segments[0]
    for seg in segments[1:]:
        result = result.unionByName(seg)
    return result.orderBy(F.rand(seed=random_state))


def load_splits_spark(spark, base_path, resample_train=True, target_samples_per_class=8000):
    """Convenience wrapper mirroring `data_pipeline.load_splits`."""
    df = create_metadata_df_spark(spark, base_path)
    train_df, val_df, test_df = patient_level_split_spark(df)
    if resample_train:
        train_df = hybrid_resample_spark(train_df, target_samples_per_class=target_samples_per_class)
    return train_df, val_df, test_df
