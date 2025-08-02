# Churn Prediction FastAPI Platform

A modular FastAPI web service for tabular customer churn prediction–including data upload, cleaning, model training (Logistic Regression, Random Forest, XGBoost), batch inference, and GenAI-powered explanations.

---

## Features

- Upload CSV data with custom target column  
- Flexible preprocessing: missing value handling, categorical encoding, scaling  
- Train multiple models: Logistic Regression, Random Forest, XGBoost  
- Batch predictions on new data  
- GenAI explanations for every prediction (OpenAI GPT-based)  
- Structured, modular, extensible Python codebase  

---

## Project Structure

```bash
churn_fastapi
├── .venv/
├── data/
├── genai/
│   └── summarizer.py
├── ml/
│   ├── inference.py
│   ├── preprocessing.py
│   ├── training.py
└── models/ 
├── main.py
├── Readme.md
└── requirements.txt
```

---

## Setup Instructions

### Clone the repository

```bash
git clone git clone https://github.com/Utsavavaiya/churn_fastapi.git
cd churn_fastapi
```

### Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set your OpenAI API key after creating a `.env` file

```env
OPENAI_API_KEY=sk-...
```

### Start the FastAPI server

```bash
uvicorn main:app --reload
```

**Or if in a subfolder:**

```bash
uvicorn app.main:app --reload
```

---

## Open the API Docs

Visit:  
**http://localhost:8000/docs**

---

## Usage

### 1. Upload your dataset

**POST** `/upload`

- Upload a CSV  
- Provide the target column name (e.g., `"Churn"`) as a form field  
- Returns dataset info and a `dataset_id` for following steps.

---

### 2. Clean & preprocess data

**POST** `/cleaning_data`  
Supply your `dataset_id` and desired strategies:

- **missing_value_strategy**: `drop`, `mean`, `median`, `mode`, etc.  
- **categorical_encoding**: `label`, `onehot`, etc.  
- **scaling**: `standard`, `minmax`, `none`

**Example:**

- `dataset_id` = `9c0ba689-9295-44e2-9241-cc8c34e820c1`  
- `missing_value_strategy` = `mode`  
- `categorical_encoding` = `onehot`  
- `scaling` = `standard`

Returns a preview of the cleaned data.

---

### 3. Train models

**POST** `/train_models`  
Supply the cleaned `dataset_id` and a comma-separated list of model names:

**Example:**  
`model_names=logistic_regression,random_forest,xgboost`

Returns metric summary (**accuracy**, **precision**, **recall**, **f1**), and filenames for each model.

---

### 4. Predict on new data

**POST** `/predict`

Input:
- A test CSV file  
- `model_filename` (from training step 3)  
- `dataset_id` (for keeping preprocessing consistent)

Returns:
- Table with original row  
- Prediction  
- Probability  
- A natural-language GenAI explanation for every customer  

---

## GenAI Explanation

Uses OpenAI GPT to generate concise user-facing explanations per prediction.

Prompt structure and format can be customized in `summarizer.py`.

### GenAI Prompt Example

The GenAI explanations are based on prompts such as:

```text
Given the following customer data, explain why this customer is predicted to churn with 92.1% confidence.

Customer Features:
- Tenure (months): 4
- Monthly Charges: 100
...

Prediction: Churn

Provide a concise, business-friendly explanation in 1-2 sentences focusing on the most important factors that led to this prediction. Use language that a business manager would understand.
```

---

## Example API Workflow

### Upload

```bash
curl -X POST "http://localhost:8000/upload" \
     -F "file=@data/mydata.csv" \
     -F "target_col=Churn"
```

### Clean

```bash
curl -X POST "http://localhost:8000/cleaning_data" \
     -F "dataset_id=xxxx" \
     -F "missing_value_strategy=mode" \
     -F "categorical_encoding=onehot" \
     -F "scaling=standard"
```

### Train

```bash
curl -X POST "http://localhost:8000/train_models" \
     -F "cleaned_dataset_id=xxxx" \
     -F "model_names=logistic_regression,random_forest,xgboost"
```

### Predict

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "test_file=@data/new_customers.csv" \
     -F "model_filename=logistic_regression_20250801_100000.joblib" \
     -F "dataset_id=xxxx"
```

---

## Notes

- All data and models are saved to the `data/` and `models/` folders, referenced by dataset and model filenames.  
- The API is modular and easy to extend for more preprocessing steps, models, or explanation providers.

---

## Troubleshooting

> **Ensure** your `.env` file exists and your OpenAI API key is valid.  
> **If** you add new preprocessing/model logic, update the `requirements.txt` if additional libraries are needed.  
> **For** large files or heavy loads, consider using a production server and async I/O solutions.

---
