import pandas as pd
import pathlib
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models(features_dir: str = "data/features", 
                 model_dir: str = "models") -> None:
    """
    Trains separate models for light intensity (S11, S12) and polarization.
    """
    features_path = pathlib.Path(features_dir)
    model_path = pathlib.Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed data...")
    X_train = pd.read_parquet(features_path / "X_train.parquet")
    y_train = pd.read_parquet(features_path / "y_train.parquet")
    X_test = pd.read_parquet(features_path / "X_test.parquet")
    y_test = pd.read_parquet(features_path / "y_test.parquet")

    # --- Section: S11 Intensity Model ---
    # S11 varies smoothly in log-space and is strongly dependent on particle 
    # volume and cross-sections. A fast-learning model is sufficient here.
    print("\n--- Training S11 Intensity Model ---")
    s11_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=1000,
        verbose=True
    )
    s11_model.fit(X_train.drop(columns=['composition']), y_train['log_S11'])
    joblib.dump(s11_model, model_path / "s11_model.joblib")

    # --- Section: S12 Element Model ---
    # S12 oscillates around zero and requires dedicated parameter tracking 
    # to avoid being drowned out by S11's large magnitude scale.
    print("\n--- Training Dedicated S12 Model ---")
    s12_model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=1500,
        verbose=True
    )
    s12_model.fit(X_train.drop(columns=['composition']), y_train['log_S12'])
    joblib.dump(s12_model, model_path / "s12_model.joblib")

    # --- Section: Dedicated Polarization Model ---
    # Polarization is a ratio (-S12/S11) and is extremely sensitive to 
    # interference effects (the "fingerprint" of the dust).
    print("\n--- Training Dedicated Polarization Model ---")
    polarization_model = MLPRegressor(
        hidden_layer_sizes=(32, 16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001, # Slower learning for stability on subtle features
        random_state=42,
        max_iter=2000,
        verbose=True
    )
    polarization_model.fit(X_train.drop(columns=['composition']), y_train['polarization'])
    joblib.dump(polarization_model, model_path / "polarization_model.joblib")


    # Evaluation
    print("\n" + "="*30)
    print("TRAINING SUMMARY")
    print("="*30)
    
    X_test_numeric = X_test.drop(columns=['composition'])
    
    # S11 Eval
    y_pred_s11 = s11_model.predict(X_test_numeric)
    r2_s11 = r2_score(y_test['log_S11'], y_pred_s11)
    print(f"log_S11 R^2: {r2_s11:.4f}")
    
    # S12 Eval
    y_pred_s12 = s12_model.predict(X_test_numeric)
    r2_s12 = r2_score(y_test['log_S12'], y_pred_s12)
    print(f"log_S12 R^2: {r2_s12:.4f}")
        
    # Polarization Eval
    y_pred_pol = polarization_model.predict(X_test_numeric)
    r2_pol = r2_score(y_test['polarization'], y_pred_pol)
    print(f"polarization R^2: {r2_pol:.4f}")
    print("="*30)

if __name__ == "__main__":
    train_models()
