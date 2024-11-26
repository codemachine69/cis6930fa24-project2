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

# Initialize spaCy for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_lg")

# Function to load unredactor.tsv data
def load_data(file_path):
    """Load and clean the unredactor dataset."""
    data = pd.read_csv(
        file_path,
        sep="\t",
        names=["split", "name", "context"],
        on_bad_lines="skip"  # Skip rows with errors
    )
    return data

# Function to load IMDB reviews
def load_reviews(imdb_directory):
    """Load reviews from IMDB dataset."""
    reviews = []
    for folder in ['train', 'test']:
        for label in ['pos', 'neg']:
            path = os.path.join(imdb_directory, folder, label)
            files = glob.glob(os.path.join(path, '*.txt'))
            for file in tqdm(files, desc=f"Loading {folder}/{label}"):
                with open(file, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
    return reviews

# Function to extract PERSON entities from text
def extract_person_entities(reviews):
    """Extract names (PERSON entities) from IMDB reviews."""
    extracted = []
    for review in tqdm(reviews, desc="Extracting PERSON entities"):
        doc = nlp(review)
        for entity in doc.ents:
            if entity.label_ == "PERSON":
                extracted.append((entity.text, review))
    return extracted

# Function to generate redacted examples
def create_redacted_examples(extracted_data):
    """Replace names with redaction blocks in the context."""
    redacted = []
    for name, context in tqdm(extracted_data, desc="Generating redacted examples"):
        redacted_context = context.replace(name, "█" * len(name))
        redacted.append((name, redacted_context))
    return redacted

# Function to combine datasets
def merge_datasets(original_data, synthetic_data, limit=None):
    """Merge original and synthetic datasets into one."""
    synthetic_df = pd.DataFrame(synthetic_data, columns=["name", "context"])
    synthetic_df["split"] = "training"
    combined = pd.concat([original_data, synthetic_df], ignore_index=True)
    
    if limit:
        combined = combined.sample(n=limit, random_state=42)
    
    return combined

# Extract features from text context
def extract_features(context, redaction_length):
    """Extract features from text surrounding the redaction block."""
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

# Prepare training and validation datasets
def prepare_data(data):
    """Prepare features and labels for machine learning."""
    X, y = [], []
    for _, row in data.iterrows():
        length = len(row["name"])
        features = extract_features(row["context"], length)
        X.append(features)
        y.append(row["name"])
    return X, y

# Train a Random Forest model
def train_classifier(X_train, y_train):
    """Train a classifier using Random Forest."""
    pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Evaluate the trained model
def evaluate_classifier(model, X_val, y_val):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_val)
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted"
    )
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

# Run predictions on test data and save results
def run_predictions(test_file_path, model):
    """Generate predictions from test data and save them."""
    test_data = pd.read_csv(test_file_path, sep="\t", names=["id", "context"])
    
    X_test = []
    for _, row in tqdm(test_data.iterrows(), desc="Processing test data", total=len(test_data)):
        length = row["context"].count("█")
        features = extract_features(row["context"], length)
        X_test.append(features)
    
    predictions = model.predict(X_test)
    
    submission_df = pd.DataFrame({"id": test_data["id"], "name": predictions})
    submission_df.to_csv("output/submission.tsv", sep="\t", index=False)
    
    print("Predictions saved to submission.tsv")

# Prepare datasets and save them to files
def prepare_and_save(unredactor_file_path, imdb_directory_path, sample_size=None):
    """Prepare datasets and save them as files."""
    
    # Load datasets and extract entities
    unredactor_data = load_data(unredactor_file_path)
    
    imdb_reviews = load_reviews(imdb_directory_path)
    
    extracted_entities = extract_person_entities(imdb_reviews)
    
    synthetic_examples = create_redacted_examples(extracted_entities)
    
    combined_dataset = merge_datasets(unredactor_data, synthetic_examples, limit=sample_size)
    
    # Split into training and validation sets
    train_set = combined_dataset[combined_dataset["split"] == "training"]
    val_set = combined_dataset[combined_dataset["split"] == "validation"]
    
    # Prepare features and labels
    print("Preparing training and validation datasets...")
    
    X_train, y_train = prepare_data(train_set)
    
    X_val, y_val = prepare_data(val_set)
    
    # Save prepared data to files
    joblib.dump((X_train, y_train), "data/train_prepared.pkl")
    
    joblib.dump((X_val, y_val), "data/val_prepared.pkl")
    
# Command-line argument parsing
def parse_args():
   parser = argparse.ArgumentParser(description="Unredactor Pipeline")
   parser.add_argument("--mode", choices=["prepare", "train", "test"], required=True,
                       help="Mode: prepare (preprocess), train (train model), or test (generate predictions)")
   parser.add_argument("--test_file", type=str,
                       help="Path to the test.tsv file (required in test mode)")
   return parser.parse_args()

if __name__ == "__main__":
   args=parse_args()

   if args.mode == "prepare":
       prepare_and_save("data/unredactor.tsv", "data/aclImdb", 5000)

   elif args.mode == "train":
       print("Loading prepared datasets...")
       X_train, y_train = joblib.load("data/train_prepared.pkl")
       X_val, y_val = joblib.load("data/val_prepared.pkl")
       
       trained_model = train_classifier(X_train, y_train)
       
       joblib.dump(trained_model, "models/model.pkl")
       print("Model saved successfully.")
       
       evaluate_classifier(trained_model, X_val, y_val)

   elif args.mode == "test":
       if not args.test_file:
           raise ValueError("Please specify --test_file when running in test mode.")
       
       trained_model = joblib.load("models/model.pkl")
       
       run_predictions(args.test_file, trained_model)