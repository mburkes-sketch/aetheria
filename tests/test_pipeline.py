import unittest
import pathlib
import numpy as np
import pandas as pd
import joblib
import sys

# Ensure Skorch / unpickler can locate MyRegressorModel and LossEquation
from src.train import MyRegressorModel, LossEquation
sys.modules['__main__'].MyRegressorModel = MyRegressorModel
sys.modules['__main__'].LossEquation = LossEquation


class TestCosmicDustPipeline(unittest.TestCase):
    """
    Onboarding sanity test suite verifying data artifacts, preprocessor scalers,
    trained neural network models, and multi-model inference.
    """

    @classmethod
    def setUpClass(cls):
        cls.repo_root = pathlib.Path(__file__).resolve().parent.parent
        cls.data_processed = cls.repo_root / "data" / "processed"
        cls.models_dir = cls.repo_root / "models"

    def test_01_processed_data_files_exist(self):
        """Verify processed Parquet datasets exist and contain required physical columns."""
        dust_data_path = self.data_processed / "dust_data.parquet"
        self.assertTrue(dust_data_path.exists(), f"Missing {dust_data_path}")

        df = pd.read_parquet(dust_data_path)
        expected_cols = {
            'theta', 'S11', 'S12', 'polarization',
            'composition', 'particle_radius', 'particle_count',
            'refractive_index_real', 'refractive_index_img'
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)), f"Columns missing: {expected_cols - set(df.columns)}")
        self.assertGreater(len(df), 1000, "Dataset contains fewer rows than expected.")

    def test_02_scalers_load_and_transform(self):
        """Verify persisted feature preprocessors and target scalers load and transform data."""
        feat_prep_path = self.data_processed / "feature_preprocessor.joblib"
        target_scaler_path = self.data_processed / "target_scaler.joblib"

        self.assertTrue(feat_prep_path.exists(), f"Missing {feat_prep_path}")
        self.assertTrue(target_scaler_path.exists(), f"Missing {target_scaler_path}")

        feat_scaler = joblib.load(feat_prep_path)
        target_scaler = joblib.load(target_scaler_path)

        # Mock feature input
        dummy_input = pd.DataFrame([{
            'theta': np.radians(30.0),
            'cos_theta': np.cos(np.radians(30.0)),
            'sin_theta': np.sin(np.radians(30.0)),
            'refractive_index_real': 1.70,
            'log_refractive_index_img': -3.0,
            'log_particle_count': 3.0,
            'log_particle_radius': -1.0,
            'size_parameter': 1.93
        }])

        transformed = feat_scaler.transform(dummy_input)
        self.assertEqual(transformed.shape, (1, 8))

    def test_03_trained_models_load(self):
        """Verify all 4 trained model artifacts exist and load properly."""
        model_names = [
            "s11_model.joblib",
            "s12_model.joblib",
            "polarization_model.joblib",
            "polarization_params_model.joblib"
        ]
        for name in model_names:
            model_path = self.models_dir / name
            self.assertTrue(model_path.exists(), f"Trained model missing: {model_path}")
            model = joblib.load(model_path)
            self.assertIsNotNone(model, f"Failed to load {name}")

    def test_04_inference_outputs(self):
        """Verify multi-model forward pass produces valid physical predictions."""
        feat_scaler = joblib.load(self.data_processed / "feature_preprocessor.joblib")
        target_scaler = joblib.load(self.data_processed / "target_scaler.joblib")
        s11_model = joblib.load(self.models_dir / "s11_model.joblib")
        s12_model = joblib.load(self.models_dir / "s12_model.joblib")
        pol_model = joblib.load(self.models_dir / "polarization_model.joblib")
        eq_model = joblib.load(self.models_dir / "polarization_params_model.joblib")

        angles_rad = np.radians(np.linspace(10, 170, 17))
        df_in = pd.DataFrame({
            'theta': angles_rad,
            'cos_theta': np.cos(angles_rad),
            'sin_theta': np.sin(angles_rad),
            'refractive_index_real': 1.68,
            'log_refractive_index_img': -2.35,
            'log_particle_count': 3.0,
            'log_particle_radius': -0.7,
            'size_parameter': (2 * np.pi * 0.20) / 0.65
        })

        feat_scaled = pd.DataFrame(feat_scaler.transform(df_in), columns=df_in.columns)

        # 1. Intensity S11
        pred_log_s11 = s11_model.predict(feat_scaled)
        dummy_matrix = np.column_stack([pred_log_s11, np.zeros_like(pred_log_s11)])
        unscaled_s11 = 10 ** (target_scaler.named_transformers_['std'].inverse_transform(dummy_matrix)[:, 0])
        self.assertTrue(np.all(unscaled_s11 > 0), "Intensity S11 should be strictly positive")

        # 2. S12
        pred_arcsinh_s12 = s12_model.predict(feat_scaled)
        dummy_matrix[:, 1] = pred_arcsinh_s12
        unscaled_s12 = np.sinh(target_scaler.named_transformers_['std'].inverse_transform(dummy_matrix)[:, 1])
        self.assertEqual(len(unscaled_s12), len(angles_rad))

        # 3. Direct MLP Polarization
        pred_pol_scaled = pol_model.predict(feat_scaled)
        pred_pol = target_scaler.named_transformers_['quant'].inverse_transform(
            pd.DataFrame({'polarization': pred_pol_scaled})
        ).flatten()
        self.assertEqual(len(pred_pol), len(angles_rad))

        # 4. Equation Model
        mat_vec = feat_scaled[[
            'refractive_index_real', 'log_refractive_index_img',
            'log_particle_count', 'log_particle_radius', 'size_parameter'
        ]].iloc[:1].to_numpy(dtype=np.float32).copy()

        params = eq_model.predict(mat_vec)[0]
        self.assertEqual(len(params), 4, "Equation model should output 4 parameters (b, c1, c2, a0)")
        b, c1, c2, a0 = params
        self.assertGreater(b, 0, "Amplitude parameter b must be > 0")


if __name__ == "__main__":
    unittest.main()
