"""Verify src/data_pipeline.py (pandas) and src/spark_pipeline.py (PySpark)
produce equivalent results on the same data.

Builds a synthetic OASIS-like folder structure (same filename convention,
same 137:1-style imbalance including a 2-patient class), runs both
pipelines' metadata extraction -> patient split -> hybrid resampling, and
asserts:
  1. Metadata extraction finds the same patient_ids per category.
  2. Patient-level split assigns the *same* patient_ids to train/val/test.
  3. Post-resampling class distribution matches (same target count per class).

Run with: python scripts/verify_pipeline_parity.py
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_pipeline import CATEGORIES, create_metadata_df, patient_level_split, hybrid_resample
from spark_pipeline import get_spark, create_metadata_df_spark, patient_level_split_spark, hybrid_resample_spark


def build_synthetic_dataset(root):
    """Mimics OASIS's real imbalance shape, including a 2-patient minority class."""
    patient_counter = 0
    category_patient_counts = {
        "Non Demented": 40,
        "Very mild Dementia": 12,
        "Mild Dementia": 6,
        "Moderate Dementia": 2,
    }
    for category, n_patients in category_patient_counts.items():
        folder = os.path.join(root, category)
        os.makedirs(folder, exist_ok=True)
        for _ in range(n_patients):
            patient_id = str(1000 + patient_counter).zfill(4)
            patient_counter += 1
            for slice_id in range(3):
                filename = f"OAS1_{patient_id}_MR1_mpr-1_{100 + slice_id}.jpg"
                open(os.path.join(folder, filename), "w").close()


def patient_sets_by_category(df, patient_col="patient_id", category_col="category"):
    return {
        category: set(df[df[category_col] == category][patient_col])
        for category in CATEGORIES
    }


def spark_patient_sets_by_category(spark_df):
    rows = spark_df.select("patient_id", "category").distinct().collect()
    result = {category: set() for category in CATEGORIES}
    for row in rows:
        result[row["category"]].add(row["patient_id"])
    return result


def main():
    tmp_root = tempfile.mkdtemp(prefix="oasis_parity_")
    print(f"Building synthetic dataset at {tmp_root} ...")
    build_synthetic_dataset(tmp_root)

    failures = []

    # ── Pandas pipeline ───────────────────────────────────────────────────
    pandas_df = create_metadata_df(tmp_root)
    pandas_train, pandas_val, pandas_test = patient_level_split(pandas_df)
    pandas_train_balanced = hybrid_resample(pandas_train, target_samples_per_class=50)

    # ── Spark pipeline ────────────────────────────────────────────────────
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        spark_df = create_metadata_df_spark(spark, tmp_root)
        spark_train, spark_val, spark_test = patient_level_split_spark(spark_df)
        spark_train_balanced = hybrid_resample_spark(spark_train, target_samples_per_class=50)

        # 1. Metadata extraction: same patient_ids per category
        pandas_meta_patients = patient_sets_by_category(pandas_df)
        spark_meta_patients = spark_patient_sets_by_category(spark_df)
        for category in CATEGORIES:
            if pandas_meta_patients[category] != spark_meta_patients[category]:
                failures.append(f"Metadata mismatch for '{category}'")
        print("1. Metadata extraction patient_ids match:", "PASS" if not failures else "FAIL")

        # 2. Patient-level split: same patient_ids per split, per category
        for split_name, p_df, s_df in [
            ("train", pandas_train, spark_train),
            ("val", pandas_val, spark_val),
            ("test", pandas_test, spark_test),
        ]:
            p_sets = patient_sets_by_category(p_df)
            s_sets = spark_patient_sets_by_category(s_df)
            for category in CATEGORIES:
                if p_sets[category] != s_sets[category]:
                    failures.append(
                        f"Split mismatch in '{split_name}'/'{category}': "
                        f"pandas={sorted(p_sets[category])} spark={sorted(s_sets[category])}"
                    )
        split_failures = [f for f in failures if f.startswith("Split mismatch")]
        print("2. Patient-level split assignments match:", "PASS" if not split_failures else "FAIL")

        # 3. Post-resampling class distribution matches
        pandas_counts = pandas_train_balanced["category"].value_counts().to_dict()
        spark_counts = {
            row["category"]: row["count"]
            for row in spark_train_balanced.groupBy("category").count().collect()
        }
        for category in CATEGORIES:
            if pandas_counts.get(category) != spark_counts.get(category):
                failures.append(
                    f"Resampled count mismatch for '{category}': "
                    f"pandas={pandas_counts.get(category)} spark={spark_counts.get(category)}"
                )
        count_failures = [f for f in failures if f.startswith("Resampled count")]
        print("3. Post-resampling class distributions match:", "PASS" if not count_failures else "FAIL")

    finally:
        spark.stop()
        shutil.rmtree(tmp_root, ignore_errors=True)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("\nAll parity checks passed — pandas and Spark pipelines are equivalent.")


if __name__ == "__main__":
    main()
