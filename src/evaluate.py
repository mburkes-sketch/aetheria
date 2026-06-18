import pandas as pd
import pathlib
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Use a non-interactive backend for plotting in CLI
import matplotlib
matplotlib.use('Agg')

def custom_inverse_transform(transformer, data_scaled_df):
    """
    Manually applies inverse_transform for a ColumnTransformer with mixed scalers.
    Standard inverse_transform can fail if the input DataFrame doesn't match 
    the exact structure/names used during fitting.
    """
    # 1. Inverse Standard Scaling for log_S11 and log_S12
    std_scaler = transformer.named_transformers_['std']
    intensity_orig = std_scaler.inverse_transform(data_scaled_df[['log_S11', 'log_S12']])
    
    # 2. Inverse Quantile Transformation for polarization
    # This brings the 'normal' distribution back to the physical [-1, 1] range.
    quant_scaler = transformer.named_transformers_['quant']
    polarization_orig = quant_scaler.inverse_transform(data_scaled_df[['polarization']])
    
    # Combine back into a single array: [log_S11, log_S12, polarization]
    combined_orig = np.hstack([intensity_orig, polarization_orig])
    return combined_orig

def evaluate_models(features_dir: str = "data/features", 
                    model_dir: str = "models",
                    scaler_dir: str = "data/processed") -> None:
    """
    Loads separate models and test/holdout data, computes metrics.
    """
    features_path = pathlib.Path(features_dir)
    model_path = pathlib.Path(model_dir)
    scaler_path = pathlib.Path(scaler_dir)
    
    # Load Models
    s11_model = joblib.load(model_path / "s11_model.joblib")
    s12_model = joblib.load(model_path / "s12_model.joblib")
    polarization_model = joblib.load(model_path / "polarization_model.joblib")
    target_transformer = joblib.load(scaler_path / "target_scaler.joblib")

    def run_eval(X, y, name="TEST SET"):
        print(f"\n--- {name} Evaluation ---")
        
        # Numeric features for prediction
        X_numeric = X.drop(columns=['composition'])
        
        y_pred_s11 = s11_model.predict(X_numeric).reshape(-1, 1)
        y_pred_s12 = s12_model.predict(X_numeric).reshape(-1, 1)
        y_pred_pol = polarization_model.predict(X_numeric).reshape(-1, 1)
        
        y_pred_scaled_df = pd.DataFrame(
            np.hstack([y_pred_s11, y_pred_s12, y_pred_pol]),
            columns=['log_S11', 'log_S12', 'polarization'],
            index=X.index
        )
        
        targets = ['log_S11', 'log_S12', 'polarization']
        
        # 1. Aggregate Scores
        print("\nAggregate Scores:")
        for i, target in enumerate(targets):
            r2 = r2_score(y.iloc[:, i], y_pred_scaled_df.iloc[:, i])
            print(f"  {target} R^2: {r2:.4f}")

        # 2. Per-Composition Scores
        print("\nPer-Composition Scores:")
        compositions = X['composition'].unique()
        for comp in compositions:
            mask = X['composition'] == comp
            print(f"  > {comp}:")
            for i, target in enumerate(targets):
                # R2 can be undefined if variance is 0, handle with try-except or check
                y_true_comp = y.loc[mask].iloc[:, i]
                y_pred_comp = y_pred_scaled_df.loc[mask].iloc[:, i]
                if len(y_true_comp) > 1 and np.var(y_true_comp) > 0:
                    r2 = r2_score(y_true_comp, y_pred_comp)
                    print(f"    {target} R^2: {r2:.4f}")
                else:
                    print(f"    {target} R^2: N/A (too few samples or no variance)")

    # 1. Standard Test Set
    print("Loading test data...")
    X_test = pd.read_parquet(features_path / "X_test.parquet")
    y_test = pd.read_parquet(features_path / "y_test.parquet")
    run_eval(X_test, y_test, "TEST SET")

    # 2. Holdout Set (if exists)
    holdout_path = features_path / "X_holdout.parquet"
    if holdout_path.exists():
        print("\nLoading holdout data...")
        X_holdout = pd.read_parquet(holdout_path)
        y_holdout = pd.read_parquet(features_path / "y_holdout.parquet")
        run_eval(X_holdout, y_holdout, "HOLDOUT SET (Generalizability)")
    
    print("\n" + "="*30)
    print("EVALUATION COMPLETE")
    print("="*30)


def predict_polarization_curve(features_dir: str = "data/features", 
                                model_dir: str = "models",
                                scaler_dir: str = "data/processed",
                                results_dir: str = "results",
                                data_file: str = "data/processed/dust_data.parquet"):
    """
    Generates predicted polarization curves vs ground truth for all compositions.
    Saves to results/polarization/
    """
    scaler_path = pathlib.Path(scaler_dir)
    model_path = pathlib.Path(model_dir)
    results_path = pathlib.Path(results_dir) / "polarization"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load Models & Scalers
    pol_model = joblib.load(model_path / "polarization_model.joblib")
    feature_preprocessor = joblib.load(scaler_path / "feature_preprocessor.joblib")
    target_transformer = joblib.load(scaler_path / "target_scaler.joblib")

    df = pd.read_parquet(data_file)
    
    # Precompute features
    wavelength = 0.65
    df['size_parameter'] = (2 * np.pi * df['particle_radius']) / wavelength
    df['log_particle_radius'] = np.log10(df['particle_radius'] + 1e-9)
    df['log_particle_count'] = np.log10(df['particle_count'] + 1e-9)

    unique_compositions = df['composition'].unique()

    for comp in unique_compositions:
        comp_df = df[df['composition'] == comp]
        max_radius = comp_df['particle_radius'].max()
        sample_comp = comp_df[comp_df['particle_radius'] == max_radius].iloc[0]
        
        theta_range_deg = np.linspace(0, 180, 181)
        theta_range_rad = np.radians(theta_range_deg)
        
        curve_df = pd.DataFrame({
            'theta': theta_range_rad,
            'cos_theta': np.cos(theta_range_rad),
            'sin_theta': np.sin(theta_range_rad),
            'refractive_index_real': sample_comp['refractive_index_real'],
            'refractive_index_img': sample_comp['refractive_index_img'],
            'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
            'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
            'size_parameter': sample_comp['size_parameter']
        })

        X_curve_scaled = feature_preprocessor.transform(curve_df)
        y_pol_scaled = pol_model.predict(X_curve_scaled)
        
        scaled_preds_df = pd.DataFrame({
            'log_S11': np.zeros(len(y_pol_scaled)),
            'log_S12': np.zeros(len(y_pol_scaled)),
            'polarization': y_pol_scaled
        })
        y_pred_orig = custom_inverse_transform(target_transformer, scaled_preds_df)
        polarization_pred = y_pred_orig[:, 2]

        mask = (df['refractive_index_real'] == sample_comp['refractive_index_real']) & \
               (df['refractive_index_img'] == sample_comp['refractive_index_img']) & \
               (df['particle_radius'] == sample_comp['particle_radius']) & \
               (df['particle_count'] == sample_comp['particle_count'])
        gt_data = df[mask].sort_values('theta')

        plt.figure(figsize=(10, 6))
        plt.plot(theta_range_deg, polarization_pred, label='ANN Prediction', color='#1f77b4', linewidth=2.5)
        if not gt_data.empty:
            plt.scatter(np.degrees(gt_data['theta']), gt_data['polarization'], label='Ground Truth', color='#d62728', s=20)

        plt.title(f"Polarization: {comp} (r={sample_comp['particle_radius']:.2f} um)")
        plt.xlabel("Angle (deg)")
        plt.ylabel("Polarization")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        safe_name = comp.replace(" ", "_").lower()
        plt.savefig(results_path / f"polarization_{safe_name}.png", dpi=300)
        plt.close()

def predict_S11(features_dir: str = "data/features", 
                model_dir: str = "models",
                scaler_dir: str = "data/processed",
                results_dir: str = "results",
                data_file: str = "data/processed/dust_data.parquet"):
    """
    Generates predicted S11 intensity curves vs ground truth.
    Saves to results/S11/
    """
    scaler_path = pathlib.Path(scaler_dir)
    model_path = pathlib.Path(model_dir)
    results_path = pathlib.Path(results_dir) / "S11"
    results_path.mkdir(parents=True, exist_ok=True)
    
    s11_model = joblib.load(model_path / "s11_model.joblib")
    feature_preprocessor = joblib.load(scaler_path / "feature_preprocessor.joblib")
    target_transformer = joblib.load(scaler_path / "target_scaler.joblib")
    df = pd.read_parquet(data_file)
    
    unique_compositions = df['composition'].unique()
    for comp in unique_compositions:
        comp_df = df[df['composition'] == comp]
        max_radius = comp_df['particle_radius'].max()
        sample_comp = comp_df[comp_df['particle_radius'] == max_radius].iloc[0]
        
        theta_range_deg = np.linspace(0, 180, 181)
        theta_range_rad = np.radians(theta_range_deg)
        curve_df = pd.DataFrame({
            'theta': theta_range_rad,
            'cos_theta': np.cos(theta_range_rad),
            'sin_theta': np.sin(theta_range_rad),
            'refractive_index_real': sample_comp['refractive_index_real'],
            'refractive_index_img': sample_comp['refractive_index_img'],
            'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
            'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
            'size_parameter': (2 * np.pi * sample_comp['particle_radius']) / 0.65
        })

        X_curve_scaled = feature_preprocessor.transform(curve_df)
        y_s11_scaled = s11_model.predict(X_curve_scaled)
        
        # Inverse transform
        scaled_preds_df = pd.DataFrame({
            'log_S11': y_s11_scaled,
            'log_S12': np.zeros(len(y_s11_scaled)),
            'polarization': np.zeros(len(y_s11_scaled))
        })
        y_pred_orig = custom_inverse_transform(target_transformer, scaled_preds_df)
        s11_pred = 10**y_pred_orig[:, 0] # Convert back from log10

        mask = (df['refractive_index_real'] == sample_comp['refractive_index_real']) & \
               (df['refractive_index_img'] == sample_comp['refractive_index_img']) & \
               (df['particle_radius'] == sample_comp['particle_radius']) & \
               (df['particle_count'] == sample_comp['particle_count'])
        gt_data = df[mask].sort_values('theta')

        plt.figure(figsize=(10, 6))
        plt.plot(theta_range_deg, s11_pred, label='ANN Prediction', color='#2ca02c', linewidth=2.5)
        if not gt_data.empty:
            plt.scatter(np.degrees(gt_data['theta']), gt_data['S11'], label='Ground Truth', color='#d62728', s=20)

        plt.yscale('log')
        plt.title(f"S11 Intensity: {comp} (r={sample_comp['particle_radius']:.2f} um)")
        plt.xlabel("Angle (deg)")
        plt.ylabel("S11")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        
        safe_name = comp.replace(" ", "_").lower()
        plt.savefig(results_path / f"s11_{safe_name}.png", dpi=300)
        plt.close()

def predict_S12(features_dir: str = "data/features", 
                model_dir: str = "models",
                scaler_dir: str = "data/processed",
                results_dir: str = "results",
                data_file: str = "data/processed/dust_data.parquet"):
    """
    Generates predicted S12 intensity curves vs ground truth.
    Saves to results/S12/
    """
    scaler_path = pathlib.Path(scaler_dir)
    model_path = pathlib.Path(model_dir)
    results_path = pathlib.Path(results_dir) / "S12"
    results_path.mkdir(parents=True, exist_ok=True)
    
    s12_model = joblib.load(model_path / "s12_model.joblib")
    feature_preprocessor = joblib.load(scaler_path / "feature_preprocessor.joblib")
    target_transformer = joblib.load(scaler_path / "target_scaler.joblib")
    df = pd.read_parquet(data_file)
    
    unique_compositions = df['composition'].unique()
    for comp in unique_compositions:
        comp_df = df[df['composition'] == comp]
        max_radius = comp_df['particle_radius'].max()
        sample_comp = comp_df[comp_df['particle_radius'] == max_radius].iloc[0]
        
        theta_range_deg = np.linspace(0, 180, 181)
        theta_range_rad = np.radians(theta_range_deg)
        curve_df = pd.DataFrame({
            'theta': theta_range_rad,
            'cos_theta': np.cos(theta_range_rad),
            'sin_theta': np.sin(theta_range_rad),
            'refractive_index_real': sample_comp['refractive_index_real'],
            'refractive_index_img': sample_comp['refractive_index_img'],
            'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
            'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
            'size_parameter': (2 * np.pi * sample_comp['particle_radius']) / 0.65
        })

        X_curve_scaled = feature_preprocessor.transform(curve_df)
        y_s12_scaled = s12_model.predict(X_curve_scaled)
        
        # Inverse transform
        scaled_preds_df = pd.DataFrame({
            'log_S11': np.zeros(len(y_s12_scaled)),
            'log_S12': y_s12_scaled,
            'polarization': np.zeros(len(y_s12_scaled))
        })
        y_pred_orig = custom_inverse_transform(target_transformer, scaled_preds_df)
        
        # S12 can be negative, so we undo the arcsinh transform with sinh
        log_s12_val = y_pred_orig[:, 1]
        s12_pred = np.sinh(log_s12_val)

        mask = (df['refractive_index_real'] == sample_comp['refractive_index_real']) & \
               (df['refractive_index_img'] == sample_comp['refractive_index_img']) & \
               (df['particle_radius'] == sample_comp['particle_radius']) & \
               (df['particle_count'] == sample_comp['particle_count'])
        gt_data = df[mask].sort_values('theta')

        plt.figure(figsize=(10, 6))
        plt.plot(theta_range_deg, s12_pred, label='ANN Prediction', color='#9467bd', linewidth=2.5)
        if not gt_data.empty:
            plt.scatter(np.degrees(gt_data['theta']), gt_data['S12'], label='Ground Truth', color='#d62728', s=20)
            
            # Bound y-axis to ground truth range + 5% padding
            gt_min = gt_data['S12'].min()
            gt_max = gt_data['S12'].max()
            gt_range = gt_max - gt_min if gt_max != gt_min else 1.0
            plt.ylim(gt_min - 0.05 * gt_range, gt_max + 0.05 * gt_range)

        plt.title(f"S12 Element: {comp} (r={sample_comp['particle_radius']:.2f} um)")
        plt.xlabel("Angle (deg)")
        plt.ylabel("S12")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        safe_name = comp.replace(" ", "_").lower()
        plt.savefig(results_path / f"s12_{safe_name}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    evaluate_models()
    predict_polarization_curve()
    predict_S11()
    predict_S12()
