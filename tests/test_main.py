import pytest
import pandas as pd
from src.unredactor import (
    load_unredactor_data,
    generate_redacted_data,
    combine_datasets,
    prepare_dataset,
    train_model,
)

# Mock Data Fixtures

@pytest.fixture
def mock_unredactor_data():
    """Provide mock unredactor data with Harry Potter references."""
    data = {
        "split": ["training", "validation", "training"],
        "name": ["Harry Potter", "Hermione Granger", "Ron Weasley"],
        "context": [
            "The chosen one, ██████████, faced Voldemort in the Forbidden Forest.",
            "The brightest witch, ███████████████, always had the answer.",
            "The loyal friend, ██████████, stood beside Harry until the very end."
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_synthetic_data():
    """Provide mock synthetic redacted data with Harry Potter references."""
    return [("Albus Dumbledore", "The headmaster of Hogwarts, ████████████████, was a mentor to Harry.")]

@pytest.fixture
def mock_combined_data(mock_unredactor_data, mock_synthetic_data):
    """Combine mock unredactor data with synthetic examples."""
    synthetic_df = pd.DataFrame(mock_synthetic_data, columns=["name", "context"])
    synthetic_df["split"] = "training"
    combined_data = pd.concat([mock_unredactor_data, synthetic_df], ignore_index=True)
    return combined_data

@pytest.fixture
def mock_training_features_and_labels():
    """Provide mock features and labels for training."""
    features = [{"feature1": 5}, {"feature1": 10}, {"feature1": 15}]
    labels = ["Harry Potter", "Hermione Granger", "Ron Weasley"]
    return features, labels

# Test Functions

def test_load_unredactor_data(mock_unredactor_data):
    """Verify that unredactor data is loaded correctly."""
    assert not mock_unredactor_data.empty, "Unredactor data should not be empty."
    assert list(mock_unredactor_data.columns) == ["split", "name", "context"], \
        "Unredactor data must include 'split', 'name', and 'context' columns."

def test_generate_redacted_data():
    """Check if names are properly redacted in the context."""
    input_data = [("Severus Snape", "The potions master at Hogwarts was Severus Snape.")]
    redacted_output = generate_redacted_data(input_data)
    
    assert len(redacted_output) == len(input_data), \
        "The number of redacted examples should match the input."
    
    for name, context in redacted_output:
        assert "█" * len(name) in context, \
            f"The context must contain a redaction block matching the length of '{name}'."

def test_combine_datasets(mock_unredactor_data, mock_synthetic_data):
    """Ensure datasets are combined correctly."""
    combined = combine_datasets(mock_unredactor_data, mock_synthetic_data)
    
    assert not combined.empty, "The combined dataset should not be empty."
    assert len(combined) == len(mock_unredactor_data) + len(mock_synthetic_data), \
        "The size of the combined dataset must equal the sum of its parts."

def test_prepare_dataset(mock_combined_data):
    """Validate feature and label preparation."""
    training_subset = mock_combined_data[mock_combined_data["split"] == "training"]
    
    features, labels = prepare_dataset(training_subset)
    
    assert len(features) == len(labels), \
        "Features and labels must have the same number of entries."
    
    assert isinstance(features[0], dict), \
        "Each feature set should be a dictionary."

def test_train_model(mock_training_features_and_labels):
    """Check if the model is trained successfully."""
    features, labels = mock_training_features_and_labels
    
    trained_model = train_model(features, labels)
    
    assert trained_model is not None, \
        "The model should be successfully trained and returned."