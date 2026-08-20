"""Patient-level data pipeline for the OASIS dementia severity dataset.

Extracted from notebooks/Dementia.ipynb so that every model trained on this
project (Custom CNN, YOLOv8, ...) sees the exact same train/val/test patients.
Splitting by `patient_id` instead of by image prevents data leakage where the
same patient's MRI slices end up in both train and test.
"""
import os
import re

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
    """Split by patient_id so no patient's slices leak across train/val/test.

    Ratios match notebooks/Dementia.ipynb exactly: `test_size` of patients are
    held out for test, then `val_size` of the *remaining* patients are held
    out for validation.
    """
    unique_patients = df['patient_id'].unique()

    train_pats, test_pats = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )
    train_pats, val_pats = train_test_split(
        train_pats, test_size=val_size, random_state=random_state
    )

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
