import argparse
import spacy
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import joblib
import os
import glob

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_lg")

# Load the unredactor.tsv dataset (ignore bad rows)
def load_unredactor_data(file_path):
    """Load the unredactor dataset into a DataFrame, ignoring bad rows."""
    data = pd.read_csv(
        file_path,
        sep="\t",
        names=["split", "name", "context"],
        on_bad_lines="skip"  # Skip problematic rows
    )
    return data

# Load IMDB reviews from train and test subdirectories
def load_imdb_reviews(imdb_path):
    """Load IMDB reviews from train and test directories."""
    reviews = []
    for dataset_type in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            path = os.path.join(imdb_path, dataset_type, sentiment)
            files = glob.glob(os.path.join(path, '*.txt'))
            for file in tqdm(files, desc=f"Loading {dataset_type}/{sentiment}"):
                with open(file, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
    return reviews

# Extract PERSON entities from IMDB reviews
def extract_names_from_reviews(reviews):
    """Extract PERSON entities from a list of reviews."""
    extracted_data = []
    for review in tqdm(reviews, desc="Extracting names from reviews"):
        doc = nlp(review)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                extracted_data.append((ent.text, review))
    return extracted_data

# Generate synthetic redacted data
def generate_redacted_data(extracted_data):
    """Generate synthetic redacted data by replacing names with redaction blocks."""
    redacted_examples = []
    for name, context in tqdm(extracted_data, desc="Generating redacted data"):
        redacted_context = context.replace(name, "█" * len(name))
        redacted_examples.append((name, redacted_context))
    return redacted_examples

# Combine unredactor.tsv and synthetic data
def combine_datasets(unredactor_data, synthetic_data, sample_size=None):
    """Combine unredactor.tsv data with synthetic redacted examples."""
    synthetic_df = pd.DataFrame(synthetic_data, columns=["name", "context"])
    synthetic_df["split"] = "training"  # Mark as training data
    combined_data = pd.concat([unredactor_data, synthetic_df], ignore_index=True)

    # If a sample size is provided, reduce the dataset size
    if sample_size:
        combined_data = combined_data.sample(n=sample_size, random_state=42)

    return combined_data

# Feature extraction
def extract_features(context, redaction_length):
    """Extract features from the redaction context."""
    doc = nlp(context)
    tokens = [token.text for token in doc]
    
    return {
        "redaction_length": redaction_length,
        "prev_word": tokens[tokens.index("█") - 1] if "█" in tokens else "",
        "next_word": tokens[tokens.index("█") + 1] if "█" in tokens else "",
        "context_length": len(context),
        "num_entities": len([ent for ent in doc.ents]),
        "contains_person": any(ent.label_ == "PERSON" for ent in doc.ents),
    }

# Prepare datasets for training and validation
def prepare_dataset(data):
    """Prepare features and labels for training and validation."""
    X, y = [], []
    for _, row in data.iterrows():
        redaction_length = len(row["name"])
        features = extract_features(row["context"], redaction_length)
        X.append(features)
        y.append(row["name"])
    return X, y

# Train the Random Forest model
def train_model(X_train, y_train):
    """Train a Random Forest model with feature vectorization."""
    pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Evaluate the model
def evaluate_model(model, X_val, y_val):
    """Evaluate the model and print precision, recall, and F1-score."""
    y_pred = model.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted")
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

# Run test mode to generate submission.tsv
def run_test_mode(test_file, model):
    """Run the model on the test file and generate submission.tsv."""
    test_data = pd.read_csv(test_file, sep="\t", names=["id", "context"])
    
    X_test = []
    for _, row in tqdm(test_data.iterrows(), desc="Processing test data", total=len(test_data)):
        redaction_length = row["context"].count("█")
        features = extract_features(row["context"], redaction_length)
        X_test.append(features)
    
    predicted_names = model.predict(X_test)
    
    submission = pd.DataFrame({"id": test_data["id"], "name": predicted_names})
    submission.to_csv("output/submission.tsv", sep="\t", index=False)
    print("Predictions saved to submission.tsv")

# Prepare mode to preprocess and save dataset
def run_prepare_mode(unredactor_file, imdb_path, output_file, sample_size=None):
    """Prepare and save the combined dataset."""
    # Load unredactor.tsv data
    unredactor_data = load_unredactor_data(unredactor_file)
    
    # Load IMDB reviews and extract PERSON entities
    imdb_reviews = load_imdb_reviews(imdb_path)
    extracted_data = extract_names_from_reviews(imdb_reviews)
    
    # Generate synthetic redacted examples from IMDB reviews
    synthetic_data = generate_redacted_data(extracted_data)
    
    # Combine datasets (unredactor.tsv + synthetic examples)
    combined_data = combine_datasets(unredactor_data, synthetic_data, sample_size=sample_size)
    
    # Split combined data into training and validation sets
    train_data = combined_data[combined_data["split"] == "training"]
    val_data = combined_data[combined_data["split"] == "validation"]
    
    # Prepare datasets for training and validation
    print("Preparing training and validation datasets...")
    X_train, y_train = prepare_dataset(train_data)
    X_val, y_val = prepare_dataset(val_data)
    
    # Save prepared data (features and labels) to separate files
    joblib.dump((X_train, y_train), "data/train_prepared.pkl")
    joblib.dump((X_val, y_val), "data/val_prepared.pkl")
    print(f"Prepared datasets saved to train_prepared.pkl and val_prepared.pkl")

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Unredactor Project")
    parser.add_argument("--mode", choices=["prepare", "train", "test"], required=True,
                        help="Mode: prepare (preprocess), train (train model), or test (generate predictions)")
    parser.add_argument("--test_file", type=str,
                        help="Path to the test.tsv file (required in test mode)")
    return parser.parse_args()

if __name__ == "__main__":
    
   args=parse_arguments()

   if args.mode == "prepare":
       run_prepare_mode(
           unredactor_file="data/unredactor.tsv",
           imdb_path="data/aclImdb",
           output_file="data/prepared_data.pkl",
           sample_size=5000  # Adjust sample size as needed
       )

   elif args.mode == "train":
        # Train Mode: Train and evaluate the model
        
        print("Loading prepared training and validation datasets...")
        X_train, y_train = joblib.load("data/train_prepared.pkl")
        X_val, y_val = joblib.load("data/val_prepared.pkl")
        
        # Train the Random Forest model
        model = train_model(X_train, y_train)
        
        # Save the trained model to a file
        joblib.dump(model, "models/model.pkl")
        print("Model saved to model.pkl")
        
        # Evaluate the model on validation set
        evaluate_model(model, X_val, y_val)

   elif args.mode == "test":
       if not args.test_file:
           raise ValueError("In test mode, --test_file must be specified.")
       
       print("Loading trained model from model.pkl")
       model = joblib.load("models/model.pkl")
       
       run_test_mode(args.test_file, model)