import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def handle_missing_values(df, strategy, target_col=None):
    """
    Handle missing values while protecting the target column.
    """
    df = df.copy()
    
    if strategy in ["drop"]:
        return df.dropna()
    elif strategy in ["mean", "impute_mean"]:
        # Only fill numeric columns, excluding target
        numeric_cols = df.select_dtypes(include=['number']).columns
        if target_col and target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif strategy in ["median", "impute_median"]:
        # Only fill numeric columns, excluding target
        numeric_cols = df.select_dtypes(include=['number']).columns
        if target_col and target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    elif strategy in ["mode", "impute_mode"]:
        for col in df.columns:
            # Skip target column
            if target_col and col == target_col:
                continue
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
        return df
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

def encode_categorical(df, method, target_col=None):
    """
    Encode categorical variables while protecting the target column.
    """
    df = df.copy()
    
    # Get categorical columns but exclude target column
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if target_col and target_col in cat_cols:
        cat_cols = cat_cols.drop(target_col)
    
    if len(cat_cols) == 0:
        return df  # No categorical columns to process
    
    if method == "label":
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == "onehot":
        df = pd.get_dummies(df, columns=cat_cols)
        
        # Convert boolean columns (True/False) to integers (1/0)
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
    else:
        raise ValueError(f"Unknown categorical encoding method: {method}")
    
    return df

def identify_binary_columns(df):
    """
    Identify binary columns (columns with only 0s and 1s).
    These are typically one-hot encoded features that shouldn't be scaled.
    """
    binary_cols = []
    for col in df.select_dtypes(include=['number']).columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1}) or unique_vals.issubset({0.0, 1.0}):
            binary_cols.append(col)
    return binary_cols

def scale_numerical(df, method, target_col=None):
    """
    Scale numerical variables while protecting the target column AND binary/one-hot encoded columns.
    """
    df = df.copy()
    
    # Get all numerical columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Remove target column
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    
    # Remove binary/one-hot encoded columns (they shouldn't be scaled)
    binary_cols = identify_binary_columns(df)
    cols_to_scale = [col for col in num_cols if col not in binary_cols]
    
    if len(cols_to_scale) == 0:
        print("No continuous numerical columns found to scale.")
        return df
    
    print(f"Scaling {len(cols_to_scale)} continuous columns: {cols_to_scale}")
    print(f"Keeping {len(binary_cols)} binary columns unscaled: {binary_cols}")
    
    if method == "standard":
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    elif method == "minmax":
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    elif method == "none":
        return df
    else:
        raise ValueError(f"Unknown numerical scaling method: {method}")
    
    return df

def clean_data(df, config, target_col=None):
    """
    Clean the data while preserving the target column and binary features.
    """
    print(f"Starting data cleaning. Target column: {target_col}")
    print(f"Original dataset shape: {df.shape}")
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df, config["missing_value_strategy"], target_col)
    print(f"After missing value handling: {df_clean.shape}")
    
    # Step 2: Encode categorical variables
    df_clean = encode_categorical(df_clean, config["categorical_encoding"], target_col)
    print(f"After categorical encoding: {df_clean.shape}")
    
    # Step 3: Scale numerical variables (excluding binary features)
    df_clean = scale_numerical(df_clean, config["scaling"], target_col)
    print(f"After numerical scaling: {df_clean.shape}")
    
    # Verify target column is unchanged
    if target_col and target_col in df_clean.columns:
        print(f"Target column '{target_col}' preserved with unique values: {sorted(df_clean[target_col].unique())}")
        print(f"Target column dtype: {df_clean[target_col].dtype}")
    
    return df_clean
