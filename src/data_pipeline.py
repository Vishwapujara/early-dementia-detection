"""Patient-level data pipeline for the OASIS dementia severity dataset.

Extracted from the original benchmark notebook so that every model trained
on this project (Custom CNN, YOLOv8, ...) sees the exact same train/val/test
patients. Splitting by `patient_id` instead of by image prevents data
leakage where the same patient's MRI slices end up in both train and test.
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# OASIS folder name (as extracted from the dataset zip) -> ordinal severity label
CATEGORIES = {
    'Non Demented': 0,
    'Very mild Dementia': 1,
    'Mild Dementia': 2,
    'Moderate Dementia': 3,
}

# OASIS filename convention: OAS1_0001_MR1_mpr-1_150.jpg
_FILENAME_PATTERN = re.compile(r'OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+)\.jpg')


def create_metadata_df(base_path):
    """Walk `base_path/<category>/` folders and build one row per MRI slice.

    Each row records the patient_id and slice_id parsed from the filename,
    so downstream splitting can operate at the patient level.
    """
    rows = []
    for folder, label in CATEGORIES.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            match = _FILENAME_PATTERN.match(filename)
            if match:
                rows.append({
                    'path': os.path.join(folder_path, filename),
                    'patient_id': match.group(1),
                    'slice_id': int(match.group(4)),
                    'label': label,
                    'category': folder,
                })
    return pd.DataFrame(rows)


def patient_level_split(df, test_size=0.20, val_size=0.10, random_state=42):
    """Split by patient_id, stratified per category, so every class has at
    least one patient in every split.

    A single pooled `train_test_split` across all patients can leave a
    severely underrepresented class (e.g. only 2 total patients for
    Moderate Dementia) with zero patients in val or test. We instead split
    each category's patients independently:
      - <= 2 patients: not enough to split three ways, so every patient is
        used in ALL three splits. Unavoidable given the data — the class is
        heavily oversampled during training anyway (see `hybrid_resample`).
      - < 6 patients: exactly 1 patient goes to test, 1 to val, the rest
        to train.
      - otherwise: the normal `test_size`/`val_size` patient-level split.

    Patient IDs are sorted before splitting so the result is deterministic
    given `random_state`, independent of filesystem/OS directory-listing
    order. This is what lets `spark_pipeline`'s split be verified as
    identical to this one (see `scripts/verify_pipeline_parity.py`).
    """
    train_pats, val_pats, test_pats = [], [], []

    for category in CATEGORIES:
        cat_patients = sorted(df[df['category'] == category]['patient_id'].unique())
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

    train_df = df[df['patient_id'].isin(train_pats)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_pats)].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_pats)].reset_index(drop=True)
    return train_df, val_df, test_df


def hybrid_resample(df, target_samples_per_class=8000, random_state=42):
    """Under-sample majority / over-sample minority classes to a target size.

    Intended for the TRAINING split only, to counter the ~137:1 class
    imbalance (Non Demented vs Moderate). Validation and test splits should
    be left at their natural distribution to give an unbiased generalization
    estimate.
    """
    segments = []
    for category in CATEGORIES:
        subset = df[df['category'] == category]
        replace = len(subset) < target_samples_per_class
        segments.append(
            subset.sample(target_samples_per_class, replace=replace, random_state=random_state)
        )
    return pd.concat(segments).sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_splits(base_path, resample_train=True, target_samples_per_class=8000):
    """Convenience wrapper: metadata extraction -> patient split -> (optional) resampled train."""
    df = create_metadata_df(base_path)
    train_df, val_df, test_df = patient_level_split(df)
    if resample_train:
        train_df = hybrid_resample(train_df, target_samples_per_class=target_samples_per_class)
    return train_df, val_df, test_df
