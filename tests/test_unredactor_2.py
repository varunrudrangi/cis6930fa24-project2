import pytest
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from unredactor import (
    load_imdb_data,
    preprocess_unredactor_file,
    train_improved_model,
    fine_tune_unredactor,
    predict_on_test
)

# Path to existing files
IMDB_DATASET_PATH = './aclImdb/train'
UNREDACTOR_FILE = './unredactor.tsv'
TEST_FILE = './test.tsv'
OUTPUT_FILE = './submission.tsv'

# Test for loading IMDb data
def test_load_imdb_data_existing():
    assert os.path.exists(IMDB_DATASET_PATH), "IMDB dataset path does not exist!"
    imdb_data = load_imdb_data(IMDB_DATASET_PATH)

    # Check that data was loaded and structured correctly
    assert not imdb_data.empty
    assert set(imdb_data.columns) == {'text', 'sentiment'}
    assert len(imdb_data) > 0

# Test for preprocessing unredactor file
def test_preprocess_unredactor_file_existing():
    assert os.path.exists(UNREDACTOR_FILE), "Unredactor file does not exist!"
    cleaned_file_path = preprocess_unredactor_file(UNREDACTOR_FILE)

    # Check that cleaned file was created
    assert os.path.exists(cleaned_file_path), "Cleaned file was not created!"

    # Check contents of the cleaned file
    cleaned_data = pd.read_csv(cleaned_file_path, sep='\t', header=None)
    assert cleaned_data.shape[1] == 3  # Ensure exactly 3 columns

# Test for training improved model with IMDb data
def test_train_improved_model_existing():
    imdb_data = load_imdb_data(IMDB_DATASET_PATH)
    clf = train_improved_model(imdb_data)

    # Check that the model is trained and is a pipeline
    assert isinstance(clf, Pipeline)

