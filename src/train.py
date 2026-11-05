import pandas as pd
import pathlib
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from skorch import NeuralNet
import torch.optim as optim

# --- 1. PyTorch Model ---
class MyRegressorModel(nn.Module):
    """
    Neural Network to predict parameters for the physical polarization equation:
    P(theta) = b * |sin(theta)|^c1 * |cos(theta/2)|^c2 * sin((theta - a0)/2)
    where theta is the phase angle (alpha = pi - scattering_angle).
    """
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
          # original:
            # nn.Linear(input_dim, 128),
            # nn.SiLU(),  # SiLU/Swish generally performs better than ReLU for physical models
            # nn.Linear(128, 128),
            # nn.SiLU(),
            # nn.Linear(128, 64),
            # nn.SiLU(),
            # nn.Linear(64, 4)  # Outputs raw params for: b, c1, c2, a0

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        raw_outputs = self.net(X)
            
        # Split raw network outputs into individual parameter components
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
        
        # Recombine parameters into a single tensor of shape [batch_size, 4]
        return torch.cat([b, c1, c2, a0], dim=1)


# --- 2. Custom Loss Function ---
class LossEquation(nn.Module):
    """
    Custom loss function evaluating the Mean Squared Error (MSE) directly on the reconstructed
    polarization curve using the physical equation and masking out dummy padding elements.
    """
    def forward(self, y_pred_params, y_true_dict):
        theta = y_true_dict['theta']
        polarization_true = y_true_dict['polarization']

        # Create mask to ignore dummy padding values (-999.0)
        mask = (theta != -999.0)

        # Convert scattering angle to phase angle: alpha = pi - theta
        theta = torch.pi - theta

        # Extract predicted parameters
        b = y_pred_params[:, 0:1]
        c1 = y_pred_params[:, 1:2]
        c2 = y_pred_params[:, 2:3]
        a0 = y_pred_params[:, 3:4]

        # Prevent c1 and c2 from exploding exponentially during initial exploration steps
        c1 = torch.clamp(c1, min=1e-5, max=15.0)
        c2 = torch.clamp(c2, min=1e-5, max=15.0)

        # Protect against ln(0) or division-by-zero gradients near boundaries (theta = 0, pi)
        safe_sin = torch.abs(torch.sin(theta)) + 1e-7
        safe_cos = torch.abs(torch.cos(theta / 2)) + 1e-7

        # Evaluate the physical curve components
        term1 = safe_sin ** c1
        term2 = safe_cos ** c2
        term3 = torch.sin((theta - a0) / 2)

        # Reconstruct the predicted polarization curve
        polarization_pred = b * term1 * term2 * term3        
        
        # Calculate standard MSE
        squared_errors = (polarization_pred - polarization_true) ** 2
            
        # Compute mean loss ONLY over the valid non-padded physical angles
        mse_loss = torch.mean(squared_errors[mask])

        # Penalty for missing the peak amplitude
        peak_true = torch.max(polarization_true, dim=1)[0]
        peak_pred = torch.max(polarization_pred, dim=1)[0]
        peak_penalty = nn.MSELoss()(peak_pred, peak_true)

        # Combine them (tune alpha based on performance, e.g., alpha = 1.0 or 2.0)
        alpha = 1.0
        total_loss = mse_loss + (alpha * peak_penalty)

        return total_loss


def train_models(features_dir: str = "data/features", 
                 model_dir: str = "models") -> None:
    """
    Trains standard MLP Regressors for light intensity (S11, S12, polarization)
    and fits the analytical physical equation model using Skorch.
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
    print("\n--- Training S11 Intensity Model ---")
    s11_model = MLPRegressor(
        hidden_layer_sizes=(32, 16, 8),
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=1000,
        verbose=True
    )
    s11_model.fit(X_train.drop(columns=['composition']), y_train['log_S11'])
    joblib.dump(s11_model, model_path / "s11_model.joblib")

    # --- Section: S12 Element Model ---
    print("\n--- Training Dedicated S12 Model ---")
    s12_model = MLPRegressor(
        hidden_layer_sizes=(32, 16, 8),
        activation='tanh',
        solver='adam',
        random_state=42,
        max_iter=1500,
        verbose=True
    )
    s12_model.fit(X_train.drop(columns=['composition']), y_train['arcsinh_S12'])
    joblib.dump(s12_model, model_path / "s12_model.joblib")

    # --- Section: Dedicated Polarization Model ---
    print("\n--- Training Dedicated Polarization Model ---")
    polarization_model = MLPRegressor(
        hidden_layer_sizes=(32, 16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0001,
        random_state=42,
        max_iter=2000,
        verbose=True
    )
    polarization_model.fit(X_train.drop(columns=['composition']), y_train['polarization'])
    joblib.dump(polarization_model, model_path / "polarization_model.joblib")

    # --- Section: Polarization Physical Equation Model ---
    print("\n--- Training Polarization Equation Model ---")
    polarization_params_model = NeuralNet(
        module=MyRegressorModel,
        criterion=LossEquation,
        optimizer=optim.Adam,
        lr=1e-3,  # Higher learning rate for stable convergence under bounds
        max_epochs=250
    )

    print("Loading preprocessed function data...")
    X_train_f = pd.read_parquet(features_path / "X_train_f.parquet")
    y_train_f = pd.read_parquet(features_path / "y_train_f.parquet")
    X_test_f = pd.read_parquet(features_path / "X_test_f.parquet")
    y_test_f = pd.read_parquet(features_path / "y_test_f.parquet")

    # 1. Determine maximum curve length for sequence padding
    max_len_train = y_train_f['theta'].str.len().max()
    max_len_test = y_test_f['theta'].str.len().max()
    global_max_len = max(max_len_train, max_len_test)

    # 2. Define sequence padding helper
    def pad_list(x, max_length, pad_value=-999.0):
        # We use -999.0 as dummy padding since 0.0 is a physically valid angle
        return list(x) + [pad_value] * (max_length - len(x))

    # 3. Apply padding to curve lists
    for df in [y_train_f, y_test_f]:
        df['theta'] = df['theta'].apply(lambda x: pad_list(x, global_max_len, pad_value=-999.0))
        df['polarization'] = df['polarization'].apply(lambda x: pad_list(x, global_max_len, pad_value=-999.0))

    # 4. Collate into standard 2D matrices for PyTorch
    y_train_polarization_dict = {
        'theta': np.stack(y_train_f["theta"].values).astype(np.float32),
        'polarization': np.stack(y_train_f["polarization"].values).astype(np.float32),
    }
    y_test_polarization_dict = {
        'theta': np.stack(y_test_f["theta"].values).astype(np.float32),
        'polarization': np.stack(y_test_f["polarization"].values).astype(np.float32),
    }

    # Extract numerical inputs (material properties)
    features_subset = [
        'refractive_index_real', 'log_refractive_index_img', 
        'log_particle_count', 'log_particle_radius', 'size_parameter'
    ]
    X_train_f = X_train_f[features_subset].to_numpy(dtype=np.float32)
    X_test_f = X_test_f[features_subset].to_numpy(dtype=np.float32)

    polarization_params_model.fit(X_train_f, y_train_polarization_dict)
    joblib.dump(polarization_params_model, model_path / "polarization_params_model.joblib")

    # --- Section: Training Summary Evaluation ---
    print("\n" + "="*30)
    print("TRAINING SUMMARY")
    print("="*30)
    
    X_test_numeric = X_test.drop(columns=['composition'])
    
    y_pred_s11 = s11_model.predict(X_test_numeric)
    r2_s11 = r2_score(y_test['log_S11'], y_pred_s11)
    print(f"log_S11 R^2: {r2_s11:.4f}")
    
    y_pred_s12 = s12_model.predict(X_test_numeric)
    r2_s12 = r2_score(y_test['arcsinh_S12'], y_pred_s12)
    print(f"arcsinh_S12 R^2: {r2_s12:.4f}")
        
    y_pred_pol = polarization_model.predict(X_test_numeric)
    r2_pol = r2_score(y_test['polarization'], y_pred_pol)
    print(f"polarization R^2: {r2_pol:.4f}")

    # Reconstruct curves from physical equation parameters and score
    y_params_pred_pol = polarization_params_model.predict(X_test_f)
    theta_val = y_test_polarization_dict['theta']
    polarization_true = y_test_polarization_dict['polarization']
    b = y_params_pred_pol[:, 0:1]
    c1 = y_params_pred_pol[:, 1:2]
    c2 = y_params_pred_pol[:, 2:3]
    a0 = y_params_pred_pol[:, 3:4]

    theta_val = np.pi - theta_val  # Convert to phase angle
    safe_sin = np.abs(np.sin(theta_val)) + 1e-7
    safe_cos = np.abs(np.cos(theta_val / 2)) + 1e-7

    term1 = safe_sin ** c1
    term2 = safe_cos ** c2
    term3 = np.sin((theta_val - a0) / 2)
    
    polarization_pred = b * term1 * term2 * term3
    polarization_pred = np.nan_to_num(polarization_pred, nan=0.0, posinf=999.0, neginf=-999.0)
    
    # Compute R2 score ignoring sequence padding
    mask = (y_test_polarization_dict['theta'] != -999.0)
    true_valid = polarization_true[mask]
    pred_valid = polarization_pred[mask]
    
    score = r2_score(true_valid, pred_valid)
    print(f"polarization equation R^2: {score:.4f}")
    print("="*30)


if __name__ == "__main__":
    train_models()
