import pandas as pd
import joblib
import os
from ml import preprocessing

MODELS_DIR = "models"

def load_model(model_filename):
    """
    Load a trained model from the models directory.
    
    Args:
        model_filename: Name of the model file (e.g., "logistic_regression_20250801_143022.joblib")
    
    Returns:
        Loaded sklearn model
    """

    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_filename}")

    model = joblib.load(model_path)
    return model


def preprocess_test_data(test_df, cleaning_config, target_col=None):
    """
    Apply the same preprocessing steps to test data as used during training.
    
    Args:
        test_df: Test DataFrame
        cleaning_config: Same config used during training
        target_col: Target column name (if present in test data)
    
    Returns:
        Preprocessed test DataFrame
    """
    # Apply the same cleaning pipeline
    processed_df = preprocessing.clean_data(test_df, cleaning_config, target_col)
    return processed_df

def make_predictions(model, X_test):
    """
    Generate predictions and prediction probabilities.
    
    Args:
        model: Trained sklearn model
        X_test: Test features DataFrame
    
    Returns:
        predictions: Array of predicted classes
        probabilities: Array of prediction probabilities (if available)
    """
    predictions = model.predict(X_test)

    # Get prediction probabilities if available
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)

    return predictions, probabilities

def predict_batch(test_df, model_filename, cleaning_config, target_col=None):
    """
    Perform batch prediction on test data.
    
    Args:
        test_df: Test DataFrame
        model_filename: Name of trained model file
        cleaning_config: Preprocessing configuration used during training
        target_col: Target column name (for preprocessing consistency)
    
    Returns:
        Dictionary containing original data, predictions, and probabilities
    """

    # Load the trained model
    model = load_model(model_filename)

    # Store original data for output
    original_data = test_df.copy()

    # Preprocess test data
    processed_df = preprocess_test_data(test_df, cleaning_config, target_col)

    # Remove target column if it exists (for true test scenarios, it might not exist)
    if target_col and target_col in processed_df.columns:
        X_test = processed_df.drop(columns=[target_col])
    else:
        X_test = processed_df

    # Make predictions
    predictions, probabilities = make_predictions(model, X_test)

    return {
        'original_data': original_data,
        'processed_data': processed_df,
        'predictions': predictions,
        'probabilities': probabilities,
        'feature_names': list(X_test.columns)
    }