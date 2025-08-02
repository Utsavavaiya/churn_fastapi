from fastapi import FastAPI, File, Form, UploadFile
import pandas as pd
import io
import uvicorn
import os
import uuid
import json
from ml import preprocessing
from ml import training
from ml import inference
from pydantic import BaseModel
from genai.summarizer import ChurnExplanationGenerator

app = FastAPI(title="Churn Prediction Platform")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

class CleaningConfig(BaseModel):
    missing_value_strategy: str
    categorical_encoding: str
    scaling: str

@app.get("/")
def root():
    return {"message": "Welcome to the Churn Prediction Platform"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    target_col: str = Form(...)
):
    # Generate a unique filename for the dataset
    dataset_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    raw_filename = f"{dataset_id}_raw{file_ext or '.csv'}"
    raw_path = os.path.join(DATA_DIR, raw_filename)
    
    # Save uploaded file to disk
    with open(raw_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    
    # Save metadata (including target column) alongside the dataset
    metadata = {
        "dataset_id": dataset_id,
        "target_column": target_col,
        "original_filename": file.filename,
        "raw_file_path": raw_filename
    }
    metadata_filename = f"{dataset_id}_metadata.json"
    metadata_path = os.path.join(DATA_DIR, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # Load to show preview and metadata
    df = pd.read_csv(raw_path)
    data_info = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "preview": df.head().to_dict(orient="records"),
        "target_column": target_col,
        "dataset_id": dataset_id
    }
    
    return {"message": "File uploaded and saved.", "data_info": data_info}

@app.post("/cleaning_data")
async def data_cleaning(
    dataset_id: str = Form(...),
    missing_value_strategy: str = Form(...),
    categorical_encoding: str = Form(...),
    scaling: str = Form(...)
):
    # Load metadata to get target column
    metadata_filename = f"{dataset_id}_metadata.json"
    metadata_path = os.path.join(DATA_DIR, metadata_filename)
    if not os.path.exists(metadata_path):
        return {"error": "Dataset metadata not found. Please upload the dataset first."}
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    target_col = metadata["target_column"]
    
    # Load raw dataset using dataset_id
    raw_filename = f"{dataset_id}_raw.csv"
    raw_path = os.path.join(DATA_DIR, raw_filename)
    if not os.path.exists(raw_path):
        return {"error": "Raw dataset not found. Please upload first."}
    
    df = pd.read_csv(raw_path)
    cleaning_config = {
        "missing_value_strategy": missing_value_strategy,
        "categorical_encoding": categorical_encoding,
        "scaling": scaling,
    }
    
    # Pass target_col to protect it from processing
    cleaned_df = preprocessing.clean_data(df, cleaning_config, target_col)
    
    # Save cleaned dataset to new file
    cleaned_filename = f"{dataset_id}_cleaned.csv"
    cleaned_path = os.path.join(DATA_DIR, cleaned_filename)
    cleaned_df.to_csv(cleaned_path, index=False)
    
    # Update metadata with cleaning info
    metadata["cleaning_config"] = cleaning_config
    metadata["cleaned_file_path"] = cleaned_filename
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # Debug: Check target column after cleaning
    print(f"Target column '{target_col}' unique values after cleaning:", cleaned_df[target_col].unique())
    print(f"Target column dtype: {cleaned_df[target_col].dtype}")
    
    preview = cleaned_df.head().to_dict(orient="records")
    return {
        "message": "Data cleaning applied",
        "preview": preview,
        "cleaned_dataset_id": dataset_id,
        "target_column": target_col,  # Return this for user's reference
        "info": {
            "num_rows": len(cleaned_df),
            "num_cols": len(cleaned_df.columns),
            "columns": list(cleaned_df.columns)
        }
    }

@app.post("/train_models")
async def train_models(
    cleaned_dataset_id: str = Form(...),
    model_names: str = Form(...),
):
    # Load metadata to get target column
    metadata_filename = f"{cleaned_dataset_id}_metadata.json"
    metadata_path = os.path.join(DATA_DIR, metadata_filename)
    if not os.path.exists(metadata_path):
        return {"error": "Dataset metadata not found."}
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    target_col = metadata["target_column"]
    
    # Load cleaned dataset from disk
    cleaned_filename = f"{cleaned_dataset_id}_cleaned.csv"
    cleaned_path = os.path.join(DATA_DIR, cleaned_filename)
    if not os.path.exists(cleaned_path):
        return {"error": "Cleaned dataset not found. Please run cleaning step first."}
    
    df = pd.read_csv(cleaned_path)
    models_to_train = [name.strip() for name in model_names.split(",")]
    results, model_files = training.train_models(df, target_col, models_to_train)
    
    # Update metadata with training info
    metadata["trained_models"] = model_files
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    return {
        "message": "Model training completed",
        "target_column": target_col,
        "metrics": results,
        "model_files": model_files
    }

@app.post("/predict")
async def predict(
    test_file: UploadFile = File(...),
    model_filename: str = Form(...), # e.g., "logistic_regression_20250801_143022.joblib"
    dataset_id: str = Form(...)  # To get the cleaning config and target column
):
    """
    Perform batch inference with GenAI explanations.
    
    Args:
        test_file: CSV file with test data
        model_filename: Name of trained model to use
        dataset_id: Original dataset ID to get preprocessing config
    """
    try:
        # Load metadata to get cleaning config and target column
        metadata_filename = f"{dataset_id}_metadata.json"
        metadata_path = os.path.join(DATA_DIR, metadata_filename)
        if not os.path.exists(metadata_path):
            return {"error": "Dataset metadata not found."}
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        cleaning_config = metadata.get("cleaning_config")
        target_col = metadata["target_column"]
        
        if not cleaning_config:
            return {"error": "Cleaning configuration not found. Please run cleaning step first."}
        
        # Read test data
        contents = await test_file.read()
        test_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        
        # Perform batch prediction
        results = inference.predict_batch(
            test_df=test_df,
            model_filename=model_filename,
            cleaning_config=cleaning_config,
            target_col=target_col
        )
        
        # Initialize GenAI explanation generator
        
        explainer = ChurnExplanationGenerator(provider="openai")  # Set your API key in environment
        
        # Generate explanations for each prediction
        explanations = []
        predictions_list = results['predictions']
        probabilities = results['probabilities']
        original_data = results['original_data']
        
        for i in range(len(predictions_list)):
            row_data = original_data.iloc[i].to_dict()
            prediction = predictions_list[i]
            probability = probabilities[i] if probabilities is not None else None
            
            explanation = explainer.generate_explanation(
                row_data=row_data,
                prediction=prediction,
                probability=probability,
                feature_names=results['feature_names']
            )
            
            explanations.append({
                "row_index": i,
                "original_data": row_data,
                "prediction": int(prediction),
                "prediction_label": "Churn" if prediction == 1 else "No Churn",
                "confidence": float(max(probability)) if probability is not None else None,
                "explanation": explanation
            })
        
        return {
            "message": "Predictions completed successfully",
            "model_used": model_filename,
            "total_predictions": len(predictions_list),
            "results": explanations
        }
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
