import pytest
import wandb
import math
import pandas as pd
import os

# This is global so all tests are collected under the same run
run = wandb.init(project='tweets_classifying', save_code=True)

# Get the clean_data artifact
artifact = run.use_artifact('clean_data:latest')
path = artifact.get_path('clean_data.csv')
data = path.download()
df = pd.read_csv(data)


def test_dataset_size():
    # Ensure that the dataset has at least 7000 rows
    assert len(df) >= 7000

def test_target_labels():
    # Ensure that the 'target' column has only 0 and 1 as labels, excluding NaN
    actual_labels = set(df['target'].unique())

    # Check for equality excluding NaN
    assert all(math.isnan(label) or label in {0.0, 1.0} for label in actual_labels)
