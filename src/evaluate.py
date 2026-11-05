import pandas as pd
import pathlib
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.optimize import curve_fit

import torch
import torch.nn as nn

# Use a non-interactive backend for plotting in CLI
import matplotlib
matplotlib.use('Agg')


# --- 1. PyTorch Model ---
class MyRegressorModel(nn.Module):
  def __init__(self, input_dim=5, hidden_dim=32):
    super().__init__()
    self.net = nn.Sequential(
          # Restored and modified to deeper architecture to capture nonlinearities for lab data
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 4)  # Outputs raw params for: b, c1, c2, a0
        )
      
    # Smoothly forces outputs to be positive (> 0) instead of hard-clamping at 0
    self.softplus = nn.Softplus() 
    self.sigmoid = nn.Sigmoid()

  def forward(self, X):
    raw_outputs = self.net(X)
        
    # Split the raw network outputs into individual parameters
    raw_b  = raw_outputs[:, 0:1]
    raw_c1 = raw_outputs[:, 1:2]
    raw_c2 = raw_outputs[:, 2:3]
    raw_a0 = raw_outputs[:, 3:4]
    
    # 1. Amplitude (b): Range [0.01, 6.0]
    b  = 0.01 + self.sigmoid(raw_b) * 6.0
    
    # 2. Exponents (c1, c2): Range [0.1, 10.0]
    c1 = 0.1 + self.sigmoid(raw_c1) * 10.0
    c2 = 0.1 + self.sigmoid(raw_c2) * 10.0
    
    # 3. Crossover phase angle (a0): Range [0.0, 1.5] rad (up to ~86 deg)
    a0 = 0.0 + self.sigmoid(raw_a0) * 1.5
    
    # Recombine them into a single tensor of shape [batch_size, 4]
    return torch.cat([b, c1, c2, a0], dim=1)

# --- 2. Custom Loss Function ---
class LossEquation(nn.Module):
# def __init__(self, over_penalty=2.0):
#     super().__init__()
#     self.register_buffer('theta', torch.tensor(theta_values, dtype=torch.float32))

  def forward(self, y_pred_params, y_true_dict):

    theta = y_true_dict['theta']
    polarization_true = y_true_dict['polarization']

    # --- CREATE THE MASK FOR THE DUMMY VALUES ---
    mask = (theta != -999.0)

    # Convert scattering angle to phase angle: alpha = pi - theta
    theta = torch.pi - theta

    #P(theta) = b (sin theta)^(c1) * (cos theta)^(c2) * sin(theta - a0)
    b = y_pred_params[:, 0:1]
    c1 = y_pred_params[:, 1:2]
    c2 = y_pred_params[:, 2:3]
    a0 = y_pred_params[:, 3:4]

    # --- PROTECTION 1: Prevent c1 and c2 from exploding exponentially ---
    # Clamping ensures the network can't guess a ridiculous exponent like 50000.0
    c1 = torch.clamp(c1, min=1e-5, max=15.0)
    c2 = torch.clamp(c2, min=1e-5, max=15.0)

    # --- FIXED: Protect against ln(0) gradients at theta = 0 and theta = pi ---
    safe_sin = torch.abs(torch.sin(theta)) + 1e-7
    safe_cos = torch.abs(torch.cos(theta / 2)) + 1e-7


    # Calculate the components safely
    term1 = safe_sin ** c1
    term2 = safe_cos ** c2
    term3 = torch.sin((theta - a0) / 2)

    polarization_pred = b * term1 * term2 * term3        
    squared_errors = ((polarization_pred - polarization_true) / (polarization_true + 1e-5)) ** 2
        
    # This keeps only the valid physics points, ignoring the -999.0 spots
    mse_loss = torch.mean(squared_errors[mask])

    # Penalty for missing the peak amplitude
    peak_true = torch.max(polarization_true, dim=1)[0]
    peak_pred = torch.max(polarization_pred, dim=1)[0]
    peak_penalty = nn.MSELoss()(peak_pred, peak_true)

    # Combine them
    alpha = 1.0
    total_loss = mse_loss + (alpha * peak_penalty)

    # --- PROTECTION 3: Emergency Fallback ---
    # If something else somehow breaks, this prevents NaN from killing the model weights
    if torch.isnan(mse_loss) or torch.isinf(mse_loss):
        return torch.tensor(999.0, requires_grad=True, device=mse_loss.device)
        
    return total_loss

def custom_inverse_transform(transformer, data_scaled_df):
    """
    Manually applies inverse_transform for a ColumnTransformer with mixed scalers.
    Standard inverse_transform can fail if the input DataFrame doesn't match 
    the exact structure/names used during fitting.
    """
    # 1. Inverse Standard Scaling for log_S11 and arcsinh_S12
    std_scaler = transformer.named_transformers_['std']
    intensity_orig = std_scaler.inverse_transform(data_scaled_df[['log_S11', 'arcsinh_S12']])
    
    # 2. Inverse Quantile Transformation for polarization
    quant_scaler = transformer.named_transformers_['quant']
    polarization_orig = quant_scaler.inverse_transform(data_scaled_df[['polarization']])
    
    # Combine back into a single array: [log_S11, arcsinh_S12, polarization]
    combined_orig = np.hstack([intensity_orig, polarization_orig])
    return combined_orig

def _reconstruct_polarization(y_params, theta):
    """
    Reconstructs the polarization curve from predicted equation parameters.
    P(theta) = b * |sin(theta)|^c1 * |cos(theta/2)|^c2 * sin((theta - a0)/2)

    Args:
        y_params: ndarray of shape (N, 4) with columns [b, c1, c2, a0].
        theta: ndarray of shape (N, L) with scattering angles in radians.
    Returns:
        polarization_pred: ndarray of shape (N, L).
    """
    # Convert scattering angle to phase angle: alpha = pi - theta
    theta = np.pi - theta

    b  = y_params[:, 0:1]
    c1 = np.clip(y_params[:, 1:2], 1e-5, 15.0)
    c2 = np.clip(y_params[:, 2:3], 1e-5, 15.0)
    a0 = y_params[:, 3:4]

    safe_sin = np.abs(np.sin(theta)) + 1e-7
    safe_cos = np.abs(np.cos(theta / 2)) + 1e-7

    term1 = safe_sin ** c1
    term2 = safe_cos ** c2
    term3 = np.sin((theta - a0) / 2)

    polarization_pred = b * term1 * term2 * term3
    polarization_pred = np.nan_to_num(polarization_pred, nan=0.0, posinf=999.0, neginf=-999.0)
    return polarization_pred

def _fit_polarization_curve_to_points(theta_and_polarization_values):
    """
    Use a fitting model to get emperical solutions to the parameters of the curve using
    P(theta) = b * |sin(theta)|^c1 * |cos(theta/2)|^c2 * sin((theta - a0)/2)
    """
    def find_a0(theta, polarization):
      bound = len(theta)

      for i in range(bound - 1):
        if polarization.iloc[i] > 0 and polarization.iloc[i+1] < 0:
          return theta.iloc[i]

      return -999

    def polarization_model(theta, b, c1, c2, a):
        # theta must be in radians
        term1 = np.abs(np.sin(theta) + 1e-7) ** c1
        term2 = np.abs(np.cos(theta / 2) + 1e-7) ** c2
        term3 = np.sin((theta - a) / 2)
        return b * term1 * term2 * term3

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #   print(theta_and_polarization_values)

    mask = ((~theta_and_polarization_values["theta"].isna()) &
            (~theta_and_polarization_values["polarization"].isna()))

    cleaned_vals = theta_and_polarization_values[mask] 
    theta = cleaned_vals["theta"]
    theta = np.pi - theta
    polarization =cleaned_vals["polarization"]

    a0 = find_a0(theta, polarization)

    if a0 != -999:
      initial_guess = [0.5, 1.0, 1.0]

      # 4. Set physical bounds to keep the parameters realistic
      # (e.g., b cannot be negative, angles are constrained)
      param_bounds = (
          [0.0, 0.0, 0.0],  # Lower bounds for [b, c1, c2, a]
          [5.0, 5.0, 5.0]   # Upper bounds for [b, c1, c2, a]
      )

      # 5. Fit the parameters
      popt, pcov = curve_fit(
          lambda theta, b, c1, c2: polarization_model(theta, b, c1, c2, a0), 
          theta, 
          polarization, 
          p0=initial_guess, 
          bounds=param_bounds
      )

      # Output results
      b_fit, c1_fit, c2_fit = popt

      # print(f"Fitted Parameters:\n b: {b_fit:.3f}\n c1: {c1_fit:.3f}\n c2: {c2_fit:.3f}\n a: {a_fit:.3f}")

      return b_fit, c1_fit, c2_fit, a0

    else:

      # 3. Provide an initial guess [b, c1, c2, a]
      # Highly non-linear trig equations need a decent starting point to avoid local minima!
      initial_guess = [0.5, 1.0, 1.0, np.pi]

      # 4. Set physical bounds to keep the parameters realistic
      # (e.g., b cannot be negative, angles are constrained)
      param_bounds = (
          [0.0, 0.0, 0.0, -np.pi],  # Lower bounds for [b, c1, c2, a]
          [5.0, 5.0, 5.0,  np.pi]   # Upper bounds for [b, c1, c2, a]
      )

      # 5. Fit the parameters
      popt, pcov = curve_fit(
          polarization_model, 
          theta, 
          polarization, 
          p0=initial_guess, 
          bounds=param_bounds
      )

      # Output results
      b_fit, c1_fit, c2_fit, a_fit = popt

      # print(f"Fitted Parameters:\n b: {b_fit:.3f}\n c1: {c1_fit:.3f}\n c2: {c2_fit:.3f}\n a: {a_fit:.3f}")

      return popt


def _load_and_pad_function_data(features_path):
    """
    Loads the collapsed (_f) parquet files and pads theta/polarization
    curves to uniform length with -999.0 sentinel values.

    Returns:
        X_train_f, y_train_dict, X_test_f, y_test_dict, global_max_len
        and optionally X_holdout_f, y_holdout_dict if holdout exists.
    """
    X_train_f = pd.read_parquet(features_path / "X_train_f.parquet")
    y_train_f = pd.read_parquet(features_path / "y_train_f.parquet")
    X_test_f  = pd.read_parquet(features_path / "X_test_f.parquet")
    y_test_f  = pd.read_parquet(features_path / "y_test_f.parquet")

    all_dfs = [y_train_f, y_test_f]

    holdout_path = features_path / "X_holdout_f.parquet"
    X_holdout_f, y_holdout_f = None, None
    if holdout_path.exists():
        X_holdout_f = pd.read_parquet(holdout_path)
        y_holdout_f = pd.read_parquet(features_path / "y_holdout_f.parquet")
        all_dfs.append(y_holdout_f)

    # Compute global max curve length across all splits
    global_max_len = max(df['theta'].str.len().max() for df in all_dfs)

    def pad_list(x, max_length, pad_value=-999.0):
        return list(x) + [pad_value] * (max_length - len(x))

    def build_dict(y_df):
        y_df = y_df.copy()
        y_df['theta'] = y_df['theta'].apply(lambda x: pad_list(x, global_max_len))
        y_df['polarization'] = y_df['polarization'].apply(lambda x: pad_list(x, global_max_len))
        return {
            'theta': np.stack(y_df['theta'].values).astype(np.float32),
            'polarization': np.stack(y_df['polarization'].values).astype(np.float32),
        }

    feature_cols = [
        'refractive_index_real', 'log_refractive_index_img',
        'log_particle_count', 'log_particle_radius', 'size_parameter'
    ]

    result = {
        'X_train_f': X_train_f[feature_cols].to_numpy(dtype=np.float32),
        'y_train_dict': build_dict(y_train_f),
        'X_test_f': X_test_f[feature_cols].to_numpy(dtype=np.float32),
        'y_test_dict': build_dict(y_test_f),
        'global_max_len': global_max_len,
    }
    if X_holdout_f is not None:
        result['X_holdout_f'] = X_holdout_f[feature_cols].to_numpy(dtype=np.float32)
        result['y_holdout_dict'] = build_dict(y_holdout_f)

    return result


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
            columns=['log_S11', 'arcsinh_S12', 'polarization'],
            index=X.index
        )
        
        targets = ['log_S11', 'arcsinh_S12', 'polarization']
        
        # Compute Derived Polarization: - sinh(arcsinh_S12) / 10^(log_S11)
        y_pred_unscaled = custom_inverse_transform(target_transformer, y_pred_scaled_df)
        y_actual_unscaled = custom_inverse_transform(target_transformer, y)
        
        pred_log_s11 = y_pred_unscaled[:, 0]
        pred_arcsinh_s12 = y_pred_unscaled[:, 1]
        pred_s11 = 10**pred_log_s11
        pred_s12 = np.sinh(pred_arcsinh_s12)
        
        pred_derived_pol = - pred_s12 / (pred_s11 + 1e-9)
        actual_pol = y_actual_unscaled[:, 2]
        
        actual_pol_series = pd.Series(actual_pol, index=X.index)
        pred_derived_pol_series = pd.Series(pred_derived_pol, index=X.index).fillna(0).replace([np.inf, -np.inf], 0)

        # 1. Aggregate Scores
        print("\nAggregate Scores:")
        for i, target in enumerate(targets):
            r2 = r2_score(y.iloc[:, i], y_pred_scaled_df.iloc[:, i])
            print(f"  {target} R^2: {r2:.4f}")
        
        r2_derived_pol = r2_score(actual_pol_series, pred_derived_pol_series)
        print(f"  derived_polarization R^2: {r2_derived_pol:.4f}")

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
            
            # Print derived polarization score per composition
            y_true_pol_comp = actual_pol_series.loc[mask]
            y_pred_pol_comp = pred_derived_pol_series.loc[mask]
            if len(y_true_pol_comp) > 1 and np.var(y_true_pol_comp) > 0:
                r2_derived = r2_score(y_true_pol_comp, y_pred_pol_comp)
                print(f"    derived_polarization R^2: {r2_derived:.4f}")
            else:
                print(f"    derived_polarization R^2: N/A (too few samples or no variance)")

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


def evaluate_equation_model(features_dir: str = "data/features",
                            model_dir: str = "models") -> None:
    """
    Evaluates the Skorch polarization equation model.
    Loads the collapsed (_f) datasets, predicts equation parameters (b, c1, c2, a0),
    reconstructs polarization curves, and reports R^2 on test and holdout sets.
    """
    features_path = pathlib.Path(features_dir)
    model_path = pathlib.Path(model_dir)

    print("\n" + "="*50)
    print("POLARIZATION EQUATION MODEL EVALUATION")
    print("="*50)

    # Load model
    pol_params_model = joblib.load(model_path / "polarization_params_model.joblib")

    # Load and pad curve-level data
    data = _load_and_pad_function_data(features_path)

    def run_equation_eval(X_f, y_dict, name="TEST SET"):
        print(f"\n--- {name} (Equation Model) ---")

        theta = y_dict['theta']
        polarization_true = y_dict['polarization']
        mask = (theta != -999.0)

        # Predict equation parameters
        y_params = pol_params_model.predict(X_f)

        # Reconstruct polarization from the analytical equation
        polarization_pred = _reconstruct_polarization(y_params, theta)

        # Flatten valid (non-padded) points for aggregate R²
        true_valid = polarization_true[mask]
        pred_valid = polarization_pred[mask]

        if len(true_valid) > 1 and np.var(true_valid) > 0:
            r2 = r2_score(true_valid, pred_valid)
            mse = mean_squared_error(true_valid, pred_valid)
            mae = mean_absolute_error(true_valid, pred_valid)
            print(f"\nAggregate Scores:")
            print(f"  polarization_equation R^2:  {r2:.4f}")
            print(f"  polarization_equation MSE:  {mse:.6f}")
            print(f"  polarization_equation MAE:  {mae:.6f}")
        else:
            print("  polarization_equation R^2: N/A (insufficient data)")

        # Per-curve R² scores
        print(f"\nPer-Curve Scores ({X_f.shape[0]} curves):")
        per_curve_r2 = []
        for i in range(X_f.shape[0]):
            curve_mask = mask[i]
            t_true = polarization_true[i][curve_mask]
            t_pred = polarization_pred[i][curve_mask]
            if len(t_true) > 1 and np.var(t_true) > 0:
                r2_i = r2_score(t_true, t_pred)
                per_curve_r2.append(r2_i)
            else:
                per_curve_r2.append(np.nan)

        valid_r2 = [r for r in per_curve_r2 if not np.isnan(r)]
        if valid_r2:
            print(f"  Mean per-curve R^2:   {np.mean(valid_r2):.4f}")
            print(f"  Median per-curve R^2: {np.median(valid_r2):.4f}")
            print(f"  Min per-curve R^2:    {np.min(valid_r2):.4f}")
            print(f"  Max per-curve R^2:    {np.max(valid_r2):.4f}")

        # Print predicted params summary
        print(f"\nPredicted Parameter Ranges:")
        param_names = ['b', 'c1', 'c2', 'a0']
        for j, pname in enumerate(param_names):
            vals = y_params[:, j]
            print(f"  {pname}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    # Test set
    run_equation_eval(data['X_test_f'], data['y_test_dict'], "TEST SET")

    # Holdout set (if exists)
    if 'X_holdout_f' in data:
        run_equation_eval(data['X_holdout_f'], data['y_holdout_dict'],
                          "HOLDOUT SET (Generalizability)")

    print("\n" + "="*50)
    print("EQUATION MODEL EVALUATION COMPLETE")
    print("="*50)


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
        unique_radii = comp_df['particle_radius'].unique()
        
        for radius in unique_radii:
            sample_comp = comp_df[comp_df['particle_radius'] == radius].iloc[0]
            
            theta_range_deg = np.linspace(0, 180, 181)
            theta_range_rad = np.radians(theta_range_deg)
            
            curve_df = pd.DataFrame({
                'theta': theta_range_rad,
                'cos_theta': np.cos(theta_range_rad),
                'sin_theta': np.sin(theta_range_rad),
                'refractive_index_real': sample_comp['refractive_index_real'],
                'log_refractive_index_img': np.clip(np.log10(sample_comp['refractive_index_img'] + 1e-9), -9.0, None),
                'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
                'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
                'size_parameter': sample_comp['size_parameter']
            })
    
            X_curve_scaled = pd.DataFrame(
                feature_preprocessor.transform(curve_df),
                columns=curve_df.columns
            )
            y_pol_scaled = pol_model.predict(X_curve_scaled)
            
            scaled_preds_df = pd.DataFrame({
                'log_S11': np.zeros(len(y_pol_scaled)),
                'arcsinh_S12': np.zeros(len(y_pol_scaled)),
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
            comp_path = results_path / safe_name
            comp_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(comp_path / f"polarization_{safe_name}_r{sample_comp['particle_radius']:.2f}.png", dpi=300)
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
        unique_radii = comp_df['particle_radius'].unique()
        
        for radius in unique_radii:
            sample_comp = comp_df[comp_df['particle_radius'] == radius].iloc[0]
            
            theta_range_deg = np.linspace(0, 180, 181)
            theta_range_rad = np.radians(theta_range_deg)
            curve_df = pd.DataFrame({
                'theta': theta_range_rad,
                'cos_theta': np.cos(theta_range_rad),
                'sin_theta': np.sin(theta_range_rad),
                'refractive_index_real': sample_comp['refractive_index_real'],
                'log_refractive_index_img': np.clip(np.log10(sample_comp['refractive_index_img'] + 1e-9), -9.0, None),
                'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
                'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
                'size_parameter': (2 * np.pi * sample_comp['particle_radius']) / 0.65
            })
    
            X_curve_scaled = pd.DataFrame(
                feature_preprocessor.transform(curve_df),
                columns=curve_df.columns
            )
            y_s11_scaled = s11_model.predict(X_curve_scaled)
            
            # Inverse transform
            scaled_preds_df = pd.DataFrame({
                'log_S11': y_s11_scaled,
                'arcsinh_S12': np.zeros(len(y_s11_scaled)),
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
            comp_path = results_path / safe_name
            comp_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(comp_path / f"s11_{safe_name}_r{sample_comp['particle_radius']:.2f}.png", dpi=300)
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
        unique_radii = comp_df['particle_radius'].unique()
        
        for radius in unique_radii:
            sample_comp = comp_df[comp_df['particle_radius'] == radius].iloc[0]
            
            theta_range_deg = np.linspace(0, 180, 181)
            theta_range_rad = np.radians(theta_range_deg)
            curve_df = pd.DataFrame({
                'theta': theta_range_rad,
                'cos_theta': np.cos(theta_range_rad),
                'sin_theta': np.sin(theta_range_rad),
                'refractive_index_real': sample_comp['refractive_index_real'],
                'log_refractive_index_img': np.clip(np.log10(sample_comp['refractive_index_img'] + 1e-9), -9.0, None),
                'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
                'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
                'size_parameter': (2 * np.pi * sample_comp['particle_radius']) / 0.65
            })
    
            X_curve_scaled = pd.DataFrame(
                feature_preprocessor.transform(curve_df),
                columns=curve_df.columns
            )
            y_s12_scaled = s12_model.predict(X_curve_scaled)
            
            # Inverse transform
            scaled_preds_df = pd.DataFrame({
                'log_S11': np.zeros(len(y_s12_scaled)),
                'arcsinh_S12': y_s12_scaled,
                'polarization': np.zeros(len(y_s12_scaled))
            })
            y_pred_orig = custom_inverse_transform(target_transformer, scaled_preds_df)
            
            # S12 can be negative, so we undo the arcsinh transform with sinh
            arcsinh_s12_val = y_pred_orig[:, 1]
            s12_pred = np.sinh(arcsinh_s12_val)
    
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
            comp_path = results_path / safe_name
            comp_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(comp_path / f"s12_{safe_name}_r{sample_comp['particle_radius']:.2f}.png", dpi=300)
            plt.close()

def predict_derived_polarization_curve(features_dir: str = "data/features", 
                                       model_dir: str = "models",
                                       scaler_dir: str = "data/processed",
                                       results_dir: str = "results",
                                       data_file: str = "data/processed/dust_data.parquet"):
    """
    Generates predicted polarization curves calculated from -S12/S11 vs ground truth.
    Saves to results/derived_polarization/
    """
    scaler_path = pathlib.Path(scaler_dir)
    model_path = pathlib.Path(model_dir)
    results_path = pathlib.Path(results_dir) / "derived_polarization"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load Models & Scalers
    s11_model = joblib.load(model_path / "s11_model.joblib")
    s12_model = joblib.load(model_path / "s12_model.joblib")
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
        unique_radii = comp_df['particle_radius'].unique()
        
        for radius in unique_radii:
            sample_comp = comp_df[comp_df['particle_radius'] == radius].iloc[0]
            
            theta_range_deg = np.linspace(0, 180, 181)
            theta_range_rad = np.radians(theta_range_deg)
            
            curve_df = pd.DataFrame({
                'theta': theta_range_rad,
                'cos_theta': np.cos(theta_range_rad),
                'sin_theta': np.sin(theta_range_rad),
                'refractive_index_real': sample_comp['refractive_index_real'],
                'log_refractive_index_img': np.clip(np.log10(sample_comp['refractive_index_img'] + 1e-9), -9.0, None),
                'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
                'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
                'size_parameter': sample_comp['size_parameter']
            })
    
            X_curve_scaled = pd.DataFrame(
                feature_preprocessor.transform(curve_df),
                columns=curve_df.columns
            )
            y_s11_scaled = s11_model.predict(X_curve_scaled)
            y_s12_scaled = s12_model.predict(X_curve_scaled)
            
            scaled_preds_df = pd.DataFrame({
                'log_S11': y_s11_scaled,
                'arcsinh_S12': y_s12_scaled,
                'polarization': np.zeros(len(y_s11_scaled))
            })
            y_pred_orig = custom_inverse_transform(target_transformer, scaled_preds_df)
            
            # Calculate derived polarization: - S12 / S11
            pred_log_s11 = y_pred_orig[:, 0]
            pred_arcsinh_s12 = y_pred_orig[:, 1]
            pred_s11 = 10**pred_log_s11
            pred_s12 = np.sinh(pred_arcsinh_s12)
            
            polarization_pred = - pred_s12 / (pred_s11 + 1e-9)
    
            mask = (df['refractive_index_real'] == sample_comp['refractive_index_real']) & \
                   (df['refractive_index_img'] == sample_comp['refractive_index_img']) & \
                   (df['particle_radius'] == sample_comp['particle_radius']) & \
                   (df['particle_count'] == sample_comp['particle_count'])
            gt_data = df[mask].sort_values('theta')
    
            plt.figure(figsize=(10, 6))
            plt.plot(theta_range_deg, polarization_pred, label='Derived Prediction (-S12/S11)', color='#ff7f0e', linewidth=2.5)
            if not gt_data.empty:
                plt.scatter(np.degrees(gt_data['theta']), gt_data['polarization'], label='Ground Truth', color='#d62728', s=20)
    
            plt.title(f"Derived Polarization: {comp} (r={sample_comp['particle_radius']:.2f} um)")
            plt.xlabel("Angle (deg)")
            plt.ylabel("Polarization")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            safe_name = comp.replace(" ", "_").lower()
            comp_path = results_path / safe_name
            comp_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(comp_path / f"derived_polarization_{safe_name}_r{sample_comp['particle_radius']:.2f}.png", dpi=300)
            plt.close()

def predict_polarization_curve_from_params(features_dir: str = "data/features", 
                                model_dir: str = "models",
                                scaler_dir: str = "data/processed",
                                results_dir: str = "results",
                                data_file: str = "data/processed/dust_data.parquet"):
    """
    Generates predicted polarization curves from the equation model vs ground truth
    for all compositions. The model predicts (b, c1, c2, a0) from material features,
    then reconstructs P(theta) = b * |sin(theta)|^c1 * |cos(theta/2)|^c2 * sin((theta-a0)/2).
    Saves to results/polarization_equation/
    """
    scaler_path = pathlib.Path(scaler_dir)
    model_path = pathlib.Path(model_dir)
    results_path = pathlib.Path(results_dir) / "polarization_equation"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load Models & Scalers
    pol_model = joblib.load(model_path / "polarization_params_model.joblib")
    feature_preprocessor = joblib.load(scaler_path / "feature_preprocessor.joblib")

    df = pd.read_parquet(data_file)
    
    # Precompute features
    wavelength = 0.65
    df['size_parameter'] = (2 * np.pi * df['particle_radius']) / wavelength
    df['log_particle_radius'] = np.log10(df['particle_radius'] + 1e-9)
    df['log_particle_count'] = np.log10(df['particle_count'] + 1e-9)

    unique_compositions = df['composition'].unique()

    for comp in unique_compositions:
        comp_df = df[df['composition'] == comp]
        unique_radii = comp_df['particle_radius'].unique()
        
        for radius in unique_radii:
            sample_comp = comp_df[comp_df['particle_radius'] == radius].iloc[0]
            
            theta_range_deg = np.linspace(0, 180, 181)
            theta_range_rad = np.radians(theta_range_deg)
            
            # Build a single-row input with the material features for this composition.
            # The equation model only needs material properties, not per-angle features.
            # We use the feature_preprocessor to scale consistently with training.
            curve_df = pd.DataFrame({
                'theta': theta_range_rad,
                'cos_theta': np.cos(theta_range_rad),
                'sin_theta': np.sin(theta_range_rad),
                'refractive_index_real': sample_comp['refractive_index_real'],
                'log_refractive_index_img': np.clip(np.log10(sample_comp['refractive_index_img'] + 1e-9), -9.0, None),
                'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
                'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
                'size_parameter': sample_comp['size_parameter']
            })
    
            X_curve_scaled = pd.DataFrame(
                feature_preprocessor.transform(curve_df),
                columns=curve_df.columns
            )
    
            # The equation model takes one row of material features and outputs one set of params.
            # Since all rows share the same material, just take the first row.
            eq_features = X_curve_scaled[[
                'refractive_index_real', 'log_refractive_index_img',
                'log_particle_count', 'log_particle_radius', 'size_parameter'
            ]].iloc[:1].to_numpy(dtype=np.float32)
    
            y_params = pol_model.predict(eq_features)  # shape (1, 4)
    
            # Reconstruct the full polarization curve from the single set of predicted params
            theta_2d = theta_range_rad.reshape(1, -1)  # shape (1, 181)
            polarization_pred = _reconstruct_polarization(y_params, theta_2d).flatten()
    
            # Ground truth data for this composition/particle configuration
            mask = (df['refractive_index_real'] == sample_comp['refractive_index_real']) & \
                   (df['refractive_index_img'] == sample_comp['refractive_index_img']) & \
                   (df['particle_radius'] == sample_comp['particle_radius']) & \
                   (df['particle_count'] == sample_comp['particle_count'])
            gt_data = df[mask].sort_values('theta')
    
            b_fit, c1_fit, c2_fit, a0_fit = _fit_polarization_curve_to_points(gt_data[["theta", "polarization"]])
            polarization_fitted = _reconstruct_polarization(np.array([[b_fit, c1_fit, c2_fit, a0_fit]]), theta_2d).flatten()
    
            # Print the predicted parameters for this composition
            b, c1, c2, a0 = y_params[0]
            comp_text = comp + ":"
            fitted_text = "fitted:"
            print(f"{comp_text:<25} r={radius:.2f} b={b:.4f}, c1={c1:.4f}, c2={c2:.4f}, a0={a0:.4f}")
            print(f"{fitted_text:<25} r={radius:.2f} b={b_fit:.4f}, c1={c1_fit:.4f}, c2={c2_fit:.4f}, a0={a0_fit:.4f}\n")
    
            plt.figure(figsize=(10, 6))
            plt.plot(theta_range_deg, polarization_pred, 
                     label='Equation Prediction', color='#17becf', linewidth=2.5)
            plt.plot(theta_range_deg, polarization_fitted,
                     label='Fitted Prediction', color='#98fa80', linewidth=2.5)
            if not gt_data.empty:
                plt.scatter(np.degrees(gt_data['theta']), gt_data['polarization'], 
                            label='Ground Truth', color='#d62728', s=20)
    
            plt.title(f"Polarization (Equation): {comp} (r={sample_comp['particle_radius']:.2f} um)")
            plt.xlabel("Angle (deg)")
            plt.ylabel("Polarization")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            safe_name = comp.replace(" ", "_").lower()
            comp_path = results_path / safe_name
            comp_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(comp_path / f"polarization_eq_{safe_name}_r{sample_comp['particle_radius']:.2f}.png", dpi=300)
            plt.close()
    
    print(f"\nEquation polarization plots saved to {results_path}/")

def predict_hybrid_polarization_curve(features_dir: str = "data/features", 
                                      model_dir: str = "models",
                                      scaler_dir: str = "data/processed",
                                      results_dir: str = "results",
                                      data_file: str = "data/processed/dust_data.parquet"):
    """
    Generates predicted polarization curves by first using the MLP to predict
    polarization values, and then using scipy curve_fit to fit the physical 
    equation to those predictions.
    Saves to results/hybrid_polarization/
    """
    scaler_path = pathlib.Path(scaler_dir)
    model_path = pathlib.Path(model_dir)
    results_path = pathlib.Path(results_dir) / "hybrid_polarization"
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
        unique_radii = comp_df['particle_radius'].unique()
        
        for radius in unique_radii:
            sample_comp = comp_df[comp_df['particle_radius'] == radius].iloc[0]
            
            theta_range_deg = np.linspace(0, 180, 181)
            theta_range_rad = np.radians(theta_range_deg)
            
            curve_df = pd.DataFrame({
                'theta': theta_range_rad,
                'cos_theta': np.cos(theta_range_rad),
                'sin_theta': np.sin(theta_range_rad),
                'refractive_index_real': sample_comp['refractive_index_real'],
                'log_refractive_index_img': np.clip(np.log10(sample_comp['refractive_index_img'] + 1e-9), -9.0, None),
                'log_particle_count': np.log10(sample_comp['particle_count'] + 1e-9),
                'log_particle_radius': np.log10(sample_comp['particle_radius'] + 1e-9),
                'size_parameter': sample_comp['size_parameter']
            })

            X_curve_scaled = pd.DataFrame(
                feature_preprocessor.transform(curve_df),
                columns=curve_df.columns
            )
            y_pol_scaled = pol_model.predict(X_curve_scaled)
            
            scaled_preds_df = pd.DataFrame({
                'log_S11': np.zeros(len(y_pol_scaled)),
                'arcsinh_S12': np.zeros(len(y_pol_scaled)),
                'polarization': y_pol_scaled
            })
            y_pred_orig = custom_inverse_transform(target_transformer, scaled_preds_df)
            mlp_polarization_pred = y_pred_orig[:, 2]

            # Fit the equation to the MLP predictions
            pred_for_fit_df = pd.DataFrame({
                "theta": theta_range_rad,
                "polarization": mlp_polarization_pred
            })
            
            try:
                b_fit, c1_fit, c2_fit, a0_fit = _fit_polarization_curve_to_points(pred_for_fit_df)
                theta_2d = theta_range_rad.reshape(1, -1)
                hybrid_polarization_pred = _reconstruct_polarization(np.array([[b_fit, c1_fit, c2_fit, a0_fit]]), theta_2d).flatten()
            except Exception as e:
                print(f"Failed to fit curve for {comp} r={radius}: {e}")
                hybrid_polarization_pred = mlp_polarization_pred # Fallback to MLP if fit fails

            mask = (df['refractive_index_real'] == sample_comp['refractive_index_real']) & \
                   (df['refractive_index_img'] == sample_comp['refractive_index_img']) & \
                   (df['particle_radius'] == sample_comp['particle_radius']) & \
                   (df['particle_count'] == sample_comp['particle_count'])
            gt_data = df[mask].sort_values('theta')

            plt.figure(figsize=(10, 6))
            
            # Plot MLP predictions
            plt.plot(theta_range_deg, mlp_polarization_pred, label='MLP Prediction', color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Plot Hybrid predictions
            plt.plot(theta_range_deg, hybrid_polarization_pred, label='Hybrid Equation Fit', color='#1f77b4', linewidth=2.5)
            
            if not gt_data.empty:
                plt.scatter(np.degrees(gt_data['theta']), gt_data['polarization'], label='Ground Truth', color='#d62728', s=20)

            plt.title(f"Hybrid Polarization: {comp} (r={sample_comp['particle_radius']:.2f} um)")
            plt.xlabel("Angle (deg)")
            plt.ylabel("Polarization")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            safe_name = comp.replace(" ", "_").lower()
            comp_path = results_path / safe_name
            comp_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(comp_path / f"hybrid_polarization_{safe_name}_r{sample_comp['particle_radius']:.2f}.png", dpi=300)
            plt.close()
            
    print(f"\nHybrid polarization plots saved to {results_path}/")

if __name__ == "__main__":
    evaluate_models()
    evaluate_equation_model()
    predict_polarization_curve()
    predict_derived_polarization_curve()
    predict_polarization_curve_from_params()
    predict_hybrid_polarization_curve()
    predict_S11()
    predict_S12()
