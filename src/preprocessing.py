import pandas as pd
import numpy as np
import pathlib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(input_file: str = "data/processed/dust_data.parquet", 
                    output_dir: str = "data/features",
                    scaler_dir: str = "data/processed") -> None:
    """
    Loads processed data, applies transformations, splits into train/test,
    scales numeric features/targets, and saves the results.
    """
    input_path = pathlib.Path(input_file)
    output_path = pathlib.Path(output_dir)
    scaler_path = pathlib.Path(scaler_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    scaler_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_path)

    # --- Section: Physics Informed Processing ---
    # This section is currently empty as per requirements.
    # Future physics-based features or transformations go here.
    # ---------------------------------------------

    # Define features and targets
    # Features: composition, scattering_angle, number_particles
    # Targets: S11, S12, polarization
    X = df[['composition', 'scattering_angle', 'number_particles']]
    y = df[['S11', 'S12', 'polarization']]

    # Split the data BEFORE scaling to prevent data leakage
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling and Encoding
    # Composition: One-Hot Encoding
    # Numeric features: Standard Scaling
    numeric_features = ['scattering_angle', 'number_particles']
    categorical_features = ['composition']

    # Question: is it a good idea to scale scattering_angle?
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ]
    )

    print("Fitting preprocessor and scaling features...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Get feature names for the scaled data
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)

    # Convert back to DataFrames for parquet storage
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Scale targets
    print("Scaling targets...")
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)

    y_train_df = pd.DataFrame(y_train_scaled, columns=y.columns)
    y_test_df = pd.DataFrame(y_test_scaled, columns=y.columns)

    # Save data to parquet
    print(f"Saving features to {output_dir}...")
    X_train_df.to_parquet(output_path / "X_train.parquet")
    X_test_df.to_parquet(output_path / "X_test.parquet")
    y_train_df.to_parquet(output_path / "y_train.parquet")
    y_test_df.to_parquet(output_path / "y_test.parquet")

    # Save scalers/preprocessor for reproducibility
    print(f"Saving scalers to {scaler_dir}...")
    joblib.dump(preprocessor, scaler_path / "feature_preprocessor.joblib")
    joblib.dump(target_scaler, scaler_path / "target_scaler.joblib")

    print("Preprocessing complete.")
    print(f"Training set size: {len(X_train_df)}")
    print(f"Testing set size: {len(X_test_df)}")
    print(f"Features: {feature_names}")

if __name__ == "__main__":
    preprocess_data()
