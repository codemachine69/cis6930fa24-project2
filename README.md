# CIS6930FA24 - Project 2 - The Unredactor

## Introduction

The Unredactor project is a machine learning pipeline designed to predict redacted names in text. It uses the IMDB Large Movie Review Dataset and a custom `unredactor.tsv` dataset to train a model capable of unredacting names based on contextual clues. The pipeline supports three modes:

1. `Prepare`: Prepares the dataset by combining and processing data from unredactor.tsv and IMDB reviews.
2. `Train`: Trains a Random Forest model using the preprocessed dataset.
3. `Test`: Predicts names for redacted text in test.tsv and generates a submission.tsv file.

## Installation

To install and run this project:

1. Ensure you have Python 3.12 installed.
2. Install pipenv if you haven't already using `pip install pipenv`.
3. Clone this repository using `https://github.com/codemachine69/cis6930fa24-project2.git` and navigate to the project directory.
4. Install the required dependencies using `pipenv install -e .`

## Pipeline Workflow

1. Prepare Mode (--mode prepare):
   - Loads `unredactor.tsv` and IMDB reviews.
   - Extracts `PERSON` entities using spaCy.
   - Generates synthetic redacted examples by replacing names with redaction blocks (█).
   - Combines the original and synthetic datasets.
   - Splits data into training and validation sets.
   - Extracts features (e.g., context length, surrounding words) and labels.
   - Saves processed datasets (`train_prepared.pkl`, `val_prepared.pkl`) for reuse.
2. Train Mode (--mode train):

   - Loads preprocessed datasets (`train_prepared.pkl`, `val_prepared.pkl`).
   - Trains a Random Forest classifier using the training data.
   - Evaluates the model on validation data (precision, recall, F1-score).
   - Saves the trained model as `model.pkl`.

3. Test Mode (--mode test):
   - Loads the trained model (`model.pkl`).
   - Processes `test.tsv`, extracting features for each redacted context.
   - Predicts names for each redacted block.
   - Saves predictions in `output/submission.tsv`.

## Usage

### Prepare Mode

Preprocess datasets and save them for training:

`pipenv run python src/unredactor.py --mode prepare`

### Train Mode

Train the model using preprocessed data:

`pipenv run python src/unredactor.py --mode train`

### Test Mode

Run predictions on a test file (`test.tsv`) and generate a submission file:

`pipenv run python src/unredactor.py --mode test --test_file tests/test.tsv`

## Datasets

There are two datasets used in this project, which are included in `data/` directory:

- IMDB Large Movie Review Dataset
- unredactor.tsv

## Implementation Details

1. `load_data(file_path)`
   - Purpose: Loads the unredactor.tsv dataset into a Pandas DataFrame, skipping any bad rows.
   - Input: Path to the unredactor.tsv file.
   - Output: A DataFrame with columns split, name, and context.
2. `load_reviews(imdb_directory)`
   - Purpose: Loads all IMDB reviews (positive and negative) from train and test subdirectories.
   - Input: Path to the IMDB dataset directory.
   - Output: A list of review texts.
3. `extract_person_entities(reviews)`
   - Purpose: Extracts PERSON entities (names) from the IMDB reviews using spaCy's NER model.
   - Input: List of review texts.
   - Output: A list of tuples, where each tuple contains a name and its corresponding review.
4. `create_redacted_examples(extracted_data)`
   - Purpose: Replaces extracted names with redaction blocks (█) in their respective contexts to create synthetic redacted examples.
   - Input: List of tuples (name, context).
   - Output: A list of tuples (redacted name, redacted context).
5. `merge_datasets(original_data, synthetic_data, limit=None)`
   - Purpose: Combines the original dataset (unredactor.tsv) with synthetic redacted examples.
   - Input:
     - Original dataset as a DataFrame.
     - Synthetic dataset as a list of tuples (name, context).
     - Optional limit to reduce dataset size for faster processing.
   - Output: A combined dataset as a DataFrame.
6. `extract_features(context, redaction_length)`
   - Purpose: Extracts features from the text surrounding the redaction block (█).
   - Input:
     - Context text containing a redaction block.
     - Length of the redacted name (number of characters in █).
   - Output: A dictionary of features, including:
     - Previous word before the block (prev_word).
     - Next word after the block (next_word).
     - Context length (context_length).
     - Number of named entities in the context (num_entities).
     - Whether a PERSON entity exists in the context (contains_person).
7. `prepare_data(data)`
   - Purpose: Prepares features (X) and labels (y) for machine learning by processing each row in the dataset.
   - Input: Dataset as a DataFrame with columns name and context.
   - Output:
     - Features as a list of dictionaries (X).
     - Labels as a list of names (y).
8. `train_classifier(X_train, y_train)`
   - Purpose: Trains a Random Forest classifier using features and labels.
   - Input:
     - Training features (X_train).
     - Training labels (y_train).
   - Output: A trained Random Forest model.
9. `evaluate_classifier(model, X_val, y_val)`
   - Purpose: Evaluates the trained model on validation data using precision, recall, and F1-score metrics.
   - Input:
     - Trained model.
     - Validation features (X_val).
     - Validation labels (y_val).
   - Output: Prints evaluation metrics.
10. `run_predictions(test_file_path, model)`
    - Purpose: Runs predictions on test data and generates a submission file (submission.tsv).
    - Input:
      - Path to the test file (test.tsv).
      - Trained model.
    - Output: Saves predictions in submission.tsv.
11. `prepare_and_save(unredactor_file_path, imdb_directory_path, sample_size=None)`
    - Purpose:
      - Combines datasets from unredactor.tsv and IMDB reviews.
      - Prepares training and validation datasets by extracting features and labels.
      - Saves processed datasets to files for reuse during training.

## Tests

- `test_load_unredactor_data()`: Verifies that unredactor.tsv is loaded correctly.
- `test_generate_redacted_data`: Checks that names are replaced with redaction blocks (█).
- `test_combine_datasets()`: Ensures original and synthetic datasets are merged correctly.
- `test_prepare_dataset()`: Tests feature extraction and label preparation for training.
- `test_train_model()`: Confirms that the Random Forest model is trained successfully.

## Validation Metrics

- Precision: 0.03
- Recall: 0.03
- F1-Score: 0.03

## Submission File

The required submission `submission.tsv` is available in the directory `output`. The generated `submission.tsv` will look like this:

id name
1 Harry Potter
2 Hermione Granger
3 Ron Weasley
...

## Assumptions

1. The IMDB dataset is organized into `train/pos`, `train/neg`, `test/pos`, and `test/neg`.
2. The redaction blocks (█) in `unredactor.tsv` match the exact length of the original name.
3. The validation split is present in `unredactor.tsv`.

## Bugs and Limitations

1. Processing Time: Preparing large datasets (e.g., IMDB reviews) can take significant time.
2. Name Ambiguity: The model may struggle with ambiguous contexts where multiple names are plausible.
3. Redaction Length Dependency: The pipeline assumes that redaction blocks accurately represent name lengths; incorrect lengths may reduce accuracy.
4. Limited Generalization: The model is trained on movie reviews, which may limit its performance on other types of text.
5. The trained model is too large to be included in GitHub, hence the pipeline has to be run in gradescope in order to generate the model in `models/` directory.
