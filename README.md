# **README: The Unredactor Project**
# cis6930fa24 -- Project 2
# Name: Varun Rudrangi

## **Project Description**
The Unredactor project aims to reconstruct redacted names from textual data, such as movie reviews, where sensitive information has been replaced with redaction blocks (`█`). By leveraging Natural Language Processing (NLP) techniques and machine learning pipelines, the project predicts the most likely names to replace redacted content. This task has applications in document analysis, anonymized text recovery, and computational linguistics.

The project consists of two main components:
1. **Sentiment Analysis**: Utilizing the IMDB dataset for feature extraction and training models.
2. **Name Prediction (Unredaction)**: Fine-tuning the sentiment analysis model to predict redacted names using contextual clues from training data.

---

## **Pipeline Overview**
The pipeline processes textual data using multiple stages:
1. **Data Loading**: Load and preprocess datasets (IMDB and custom unredactor training sets).
2. **Feature Engineering**: Extract features such as sentiment polarity, word embeddings, and contextual information.
3. **Model Training**: Train classifiers using engineered features to predict sentiments or redacted names.
4. **Evaluation**: Validate model performance using precision, recall, and F1-score.
5. **Prediction**: Apply the trained model to a test set for unredacted name prediction.

---

## **How to Use**
### **Installation**
```bash
pip install pipenv
pipenv install
```
The key libraries used are:
- `nltk`
- `scikit-learn`
- `pandas`
- `numpy`

### **Execution**
Run the pipeline using the `main.py` script:
```bash
pipenv run python unredactor.py
```

### **Input Files**
- **IMDB Data**: Movie reviews dataset with `pos` and `neg` sentiment labels.
- **Unredactor Dataset** (`unredactor.tsv`): Contains `split`, `name`, and `context` columns.
- **Test Dataset** (`test.tsv`): Contains `split`, `name`, and `context` columns.

### **Output**
- A predictions file (`submission.tsv`) with two columns: `id` (test ID) and `name` (predicted name).

---

## **Detailed Pipeline Explanation**

### **1. Data Loading**
The `load_imdb_data` function reads IMDB movie review files, both `positive` and `negative`, and creates a DataFrame for training sentiment models.

#### Code:
```python
def load_imdb_data(base_path):
    ...
    return pd.DataFrame(data)
```
- **Input**: Directory path containing `pos` and `neg` subdirectories.
- **Output**: DataFrame with `text` and `sentiment` columns.

### **2. Custom Transformers**
Custom feature extractors are implemented as `TransformerMixin` classes for integration into a `FeatureUnion`.

#### **VaderSentimentTransformer**
Uses Vader Sentiment Analyzer to compute a sentiment score for each document.
```python
class VaderSentimentTransformer(BaseEstimator, TransformerMixin):
    ...
```

#### **LengthTransformer**
Generates text length-based features:
1. Character count
2. Word count
3. Average word length
```python
class LengthTransformer(BaseEstimator, TransformerMixin):
    ...
```

### **3. Pipeline Construction**
Combines multiple feature extractors and trains a `RandomForestClassifier`.

#### Code:
```python
def build_pipeline():
    ...
    return pipeline
```
- **FeatureUnion Components**:
  - `tfidf`: TF-IDF vectorizer for text embeddings.
  - `sentiment`: Sentiment score from Vader.
  - `length`: Length-based features.
- **Classifier**: Random Forest with 200 estimators.

### **4. Model Training**
Trains the pipeline using IMDB data and evaluates performance on validation data.
```python
def train_improved_model(imdb_data):
    ...
```

### **5. Data Preprocessing**
The `preprocess_unredactor_file` function cleans `unredactor.tsv` to remove invalid lines (e.g., missing fields).

#### Code:
```python
def preprocess_unredactor_file(file_path):
    ...
    return cleaned_file_path
```

### **6. Fine-Tuning**
The `fine_tune_unredactor` function retrains the model for name prediction using the unredactor dataset. This involves creating features from the `context` column and predicting the `name` column.

#### Code:
```python
def fine_tune_unredactor(clf, vectorizer, unredactor):
    ...
```
- **Input**: Pre-trained classifier and cleaned unredactor data.
- **Output**: Fine-tuned classifier for name prediction.

### **7. Prediction**
The `predict_on_test` function applies the trained model to predict names for redacted test data.
```python
def predict_on_test(clf, vectorizer, test_file, output_file):
    ...
```

---

## **Main Function**
The `main()` function serves as the entry point for the script and orchestrates the entire pipeline from data loading to generating predictions. It performs the following steps:
1. **Load IMDB Dataset**: Calls `load_imdb_data()` to load and preprocess the IMDB dataset for sentiment analysis.
2. **Train Sentiment Model**: Uses `train_improved_model()` to train the sentiment analysis pipeline.
3. **Preprocess Unredactor Data**: Cleans the `unredactor.tsv` file using `preprocess_unredactor_file()`.
4. **Fine-Tune Model**: Adapts the sentiment analysis model for name prediction with `fine_tune_unredactor()`.
5. **Generate Predictions**: Applies the fine-tuned model to the test dataset using `predict_on_test()` and saves the predictions to `submission.tsv`.

#### Code:
```python
def main():
    imdb_data = load_imdb_data('./aclImdb/train')
    clf = train_improved_model(imdb_data)
    cleaned_file_path = preprocess_unredactor_file('./unredactor.tsv')
    unredactor = pd.read_csv(cleaned_file_path, sep='\t', header=None, names=['split', 'name', 'context'])
    clf = fine_tune_unredactor(clf, None, unredactor)
    predict_on_test(clf, None, './test.tsv', './submission.tsv')
```

---
## **Evaluation**
### **Metrics**
- **Precision**: Accuracy of name predictions.
- **Recall**: Proportion of actual names correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.

### **Example Output**
For a test input file:
```plaintext
id  context
1   He was talking to ███████████ about the incident.
```
The `submission.tsv` file will contain:
```plaintext
id  name
1   John Smith
```

---

## **Bugs**
- **Name Length**: The redaction block's length corresponds to the number of characters in the redacted name.
- **Context Window**: Context size is limited to 1024 characters.
- **Resource Limitations**: Training is optimized to handle the IMDB dataset efficiently.

- **TF-IDF Vectorizer Misalignment**:
In build_pipeline(), the TfidfVectorizer extracts features, but it is not clearly tailored to the redaction context.
Features extracted may not correlate well with names in the redaction context.

- **Lack of Contextual Features**:
The pipeline does not explicitly include key features mentioned in the project, such as:
Number of characters in the redacted block.
Previous and next words around the redaction.
Improper Handling of Redacted Contexts

The redacted names are stored as single blocks (█████████), but the pipeline might not properly parse or match the redacted block length with candidate names.
Data Preprocessing Issues

- **Mismatch in Data Columns**:
The preprocess_unredactor_file() function may remove or mislabel rows during preprocessing.
- **Improper Validation Split**:
The splitting logic in train_test_split() might not align with the split structure (training, validation) in the unredactor.tsv file.
Pipeline Overfitting

The RandomForestClassifier may be overfitting to training data due to the absence of cross-validation and hyperparameter tuning.
Features in the training set might differ significantly from the validation set, causing poor generalization.
Sentiment Feature Misalignment

- **Irrelevant Sentiment Analysis**:
Including sentiment as a feature might dilute the model's focus on identifying names, especially if sentiment has no significant correlation with redacted contexts.
Prediction Missteps

- **Fine-tuning Misalignment**:
The fine_tune_unredactor() function appears to re-train the classifier but may lack feature alignment, as vectorizer is set to None.

## **Assumptions**
**Data Characteristics**

Names in unredactor.tsv are diverse, with varying lengths and cultural origins, which makes predicting based on limited features challenging.

**Limited Training Data**

The training set might not have enough examples to build a strong predictive model for unseen data in the validation set.

**Mismatch Between IMDB Dataset and Redaction Context**

The IMDB dataset used for initial training might not align with the linguistic or contextual patterns in the unredactor.tsv data.

**Imbalanced Dataset**

The training data could be imbalanced, leading the classifier to favor certain predictions disproportionately.

---

## **Testing**

## Unit Tests for Unredactor Module

This test suite ensures the functionality and reliability of the components in the `unredactor` module. Below is an explanation of each test:

### 1. `test_vader_sentiment_transformer`
**Description**:  
Tests the `VaderSentimentTransformer`, which is responsible for extracting sentiment scores from textual data. This transformer converts a list of texts into sentiment scores (positive, neutral, or negative).

**Assertions**:  
- Checks if the transformer returns a 2D array with the correct shape.
- Verifies that positive texts yield a positive sentiment score and negative texts yield a negative sentiment score.

### 2. `test_length_transformer`
**Description**:  
Tests the `LengthTransformer`, which computes various length-related features from textual data, such as character count, word count, and average word length.

**Assertions**:  
- Ensures the output array has three features for each text input (length, word count, and average word length).
- Validates that longer texts produce greater feature values compared to shorter texts.

### 3. `test_build_pipeline`
**Description**:  
Tests the `build_pipeline` function, which creates a complete machine learning pipeline for text processing and classification. The pipeline includes feature extraction and a `RandomForestClassifier`.

**Assertions**:  
- Confirms that the output is an instance of `Pipeline`.
- Ensures the pipeline includes `features` and `classifier` steps.
- Validates that the classifier in the pipeline is a `RandomForestClassifier`.

### 4. `test_train_improved_model`
**Description**:  
Tests the `train_improved_model` function, which trains a machine learning model using the provided IMDB dataset. This function integrates all components of the pipeline to create a working classifier.

**Assertions**:  
- Checks that the returned object is a pipeline.
- Ensures that the model can handle a small dataset with text and sentiment labels.

---

**Running the Tests**:  
To run these tests, use the following command in your terminal:
```bash
pipenv run python -m pytest -v
```