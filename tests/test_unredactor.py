import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Import functions and classes from the main code
from unredactor import (
    load_imdb_data,
    VaderSentimentTransformer,
    LengthTransformer,
    build_pipeline,
    train_improved_model,
    fine_tune_unredactor
)

# Test for VaderSentimentTransformer
def test_vader_sentiment_transformer():
    transformer = VaderSentimentTransformer()
    test_texts = ["I absolutely love this!", "This is terrible."]
    transformed = transformer.fit_transform(test_texts)

    assert transformed.shape == (2, 1)
    assert transformed[0, 0] > 0  # Positive sentiment
    assert transformed[1, 0] < 0  # Negative sentiment

# Test for LengthTransformer
def test_length_transformer():
    transformer = LengthTransformer()
    test_texts = ["Short text.", "A much longer piece of text here."]
    transformed = transformer.fit_transform(test_texts)

    assert transformed.shape == (2, 3)  # Length, word count, avg word length
    assert transformed[0, 0] < transformed[1, 0]  # Length of text
    assert transformed[0, 1] < transformed[1, 1]  # Word count

# Test for building pipeline
def test_build_pipeline():
    pipeline = build_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert 'features' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps
    assert isinstance(pipeline.named_steps['classifier'], RandomForestClassifier)

# Test for training improved model
def test_train_improved_model():
    imdb_data = pd.DataFrame({
        'text': ["I love this movie.", "I hate this movie."],
        'sentiment': [1, 0]
    })

    clf = train_improved_model(imdb_data)
    assert isinstance(clf, Pipeline)


