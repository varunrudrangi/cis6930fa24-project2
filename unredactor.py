import pandas as pd
import glob
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Step 1: Load IMDB Data
def load_imdb_data(base_path):
    data = []
    for label in ['pos', 'neg']:
        path = os.path.join(base_path, label, '*.txt')
        for file_path in glob.glob(path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                sentiment = 1 if label == 'pos' else 0
                data.append({'text': text, 'sentiment': sentiment})
    return pd.DataFrame(data)

# Step 2: Custom Transformers
class VaderSentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sentiments = [self.analyzer.polarity_scores(doc)['compound'] for doc in X]
        return np.array(sentiments).reshape(-1, 1)

class LengthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lengths = [[len(doc), len(doc.split()), np.mean([len(word) for word in doc.split()])] for doc in X]
        return np.array(lengths)

# Step 3: Build Enhanced Pipeline
def build_pipeline():
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    sentiment = VaderSentimentTransformer()
    length = LengthTransformer()
    
    features = FeatureUnion([
        ('tfidf', tfidf),
        ('sentiment', sentiment),
        ('length', length)
    ])
    
    pipeline = Pipeline([
        ('features', features),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    return pipeline

# Step 4: Train Enhanced Model
def train_improved_model(imdb_data):
    X = imdb_data['text']
    y = imdb_data['sentiment']

    # Train/Test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    pipeline.predict(X_val)
    return pipeline

# Step 5: Load and Preprocess Unredactor Data
def preprocess_unredactor_file(file_path):
    valid_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            fields = line.strip().split('\t')
            if len(fields) == 3:  # Keep only lines with exactly 3 fields
                valid_lines.append(line)
    cleaned_file_path = file_path.replace('.tsv', '_cleaned.tsv')
    with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
        cleaned_file.writelines(valid_lines)
    return cleaned_file_path

# Fine-tune Model with Unredactor Data
def fine_tune_unredactor(clf, vectorizer, unredactor):
    train_data = unredactor[unredactor['split'] == 'training']
    val_data = unredactor[unredactor['split'] == 'validation']

    X_train = train_data['context']
    y_train = train_data['name']
    X_val = val_data['context']
    y_val = val_data['name']

    clf.fit(X_train, y_train)

    # Evaluate on validation
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    return clf

# Step 6: Predict on Test Data
def predict_on_test(clf, vectorizer, test_file, output_file):
    test_data = pd.read_csv(test_file, sep='\t', header=None, names=['id', 'context'])
    X_test = test_data['context']
    predictions = clf.predict(X_test)

    test_data['name'] = predictions
    test_data[['id', 'name']].to_csv(output_file, index=False, sep='\t')

    print(f"Predictions saved to {output_file}")

# Main Pipeline
def main():
   
    imdb_data = load_imdb_data('./aclImdb/train')

    clf = train_improved_model(imdb_data)
    cleaned_file_path = preprocess_unredactor_file('./unredactor.tsv')
    unredactor = pd.read_csv(cleaned_file_path, sep='\t', header=None, names=['split', 'name', 'context'])
    clf = fine_tune_unredactor(clf, None, unredactor)
    predict_on_test(clf, None, './test.tsv', './submission.tsv')

if __name__ == "__main__":
    main()
