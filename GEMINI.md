# Project Context: Cosmic Dust ANN
You are an expert assistant in Astrophysics and Machine Learning. 
You are helping me build an Artificial Neural Network (ANN) to predict 
light scattering features from dust composition, particle size, and phase angle.

## Technical Environment
- **Language:** Python 3.10+
- **Key Libraries:** Pandas, NumPy, Scikit-Learn, PyTorch (or TensorFlow), h5py
- **Data Source:** .h5 files in `data/raw/`

## Repository Rules
- Follow the established modular structure (`src/`, `data/`, `models/`).
- Prioritize data reproducibility (save scalers in `data/processed/`).
- Use LaTeX for any physics or mathematical formulas.

## Completed Milestones
- **Data Ingestion**: Successfully mapped all synthetic (`.h5`), MSTM (`.dat`), and experimental (`.txt`) files to a central parquet file. 
    - **MSTM Parser**: Implemented `parse_dat_file` to handle both headered and headless `.dat` formats, extracting metadata (composition, particle count) from filenames and physical constants from headers.
    - **Normalization**: Standardized composition names (e.g., `amorphous_carbon` $\rightarrow$ `am_carbon`) and mapped scattering fields ($\theta$, $S_{11}$, $S_{12}$, and $polarization$).
    - **Calculations**: Calculated complex dielectric constants ($\epsilon = \epsilon' + i\epsilon''$) and normalized $S_{11}$ at $30^\circ$ for cross-format consistency.
    - **Data Fidelity Visualization**: Added automatic generation of diagnostic plots for each composition (S11, S12, polarization) in `data/processed/graphs/` to verify raw data quality before preprocessing.
- **Data Preprocessing & Feature Engineering**: Implemented `src/preprocessing.py`.
    - **Physics-Informed Features**: Added `size_parameter` ($x = \frac{2\pi r}{\lambda}$), `refractive_index` ($m = \sqrt{\epsilon}$), and trigonometric transforms for $\theta$ ($\cos\theta, \sin\theta$).
    - **Scaling Strategy**: Applied `StandardScaler` to most features and `QuantileTransformer` to the $polarization$ target to handle its non-Gaussian distribution.
    - **Data Splitting**: 80/20 train/test split.
- **Model Training**: Implemented separate Scikit-Learn `MLPRegressor` models to capture the different dynamics of light intensity and polarization.
- **Evaluation & Visualization**: Developed `src/evaluate.py` to calculate $R^2$ metrics on scaled and original data. 
    - **Visualization Suite**: Added tools to automatically generate and save comparison plots for predicted vs. ground-truth curves for $S11$ (intensity), $S12$, and $polarization$ across all compositions in the `results/` directory.
    - **Generalizability Testing**: Implemented a "Holdout Composition" strategy in `src/preprocessing.py`. This allows training on a subset of materials and validating performance on a completely unseen composition (e.g., `halley_dust`), providing a rigorous measure of the model's physical generalizability.
- **Target Consistency & Smoothness Optimization**:
    - **Physical Scaling Alignment**: Moved `arcsinh_S12` target scaling from `QuantileTransformer` to `StandardScaler` to prevent sensitivity spikes around zero.
    - **Smooth Oscillations**: Switched the $S_{12}$ hidden layer activation function to `tanh` to ensure smooth, non-jagged prediction curves.
    - **Derived Polarization Testing**: Added secondary metrics and visualization curves to evaluate physical polarization consistency ($P_{\text{derived}} = -S_{12}/S_{11}$).
- **Log-Scaling & Clipping for Imaginary Refractive Index**:
    - **Physical Feature Range**: Replaced linear `refractive_index_img` ($k$) with `log_refractive_index_img = np.clip(np.log10(k + 1e-9), -9.0, None)` to account for absorption spanning 8 orders of magnitude.
    - **Extrapolation Control**: Extended the minimum clip floor to $-9.0$ so near-transparent materials (e.g., water ice at $k \approx 1.43 \times 10^{-7}$ and `Sample 7_lab` at $10^{-8}$) remain distinguishable from the boundary while preventing gradient divergence.
- **Analytical Polarization Equation Modeling**:
    - **Physics-Informed Parameter Network**: Implemented `MyRegressorModel` in PyTorch (wrapped via Skorch) to predict physical parameters $(b, c_1, c_2, a_0)$ for the Rayleigh-Mie-inspired empirical phase function $P(\alpha) = b \cdot |\sin\alpha|^{c_1} \cdot |\cos(\alpha/2)|^{c_2} \cdot \sin((\alpha - a_0)/2)$.
    - **Loss with Peak Penalty**: Designed `LossEquation` with sequence padding mask ($-999.0$) and a compound peak amplitude penalty to ensure accurate reproduction of maximum polarization $P_{\max}$.
    - **Hybrid Fit Model**: Added a hybrid evaluation workflow combining direct MLP feature predictions with `scipy.optimize.curve_fit` parameter estimation.

## Current Objective
The primary focus is to bridge the performance gap between seen and unseen compositions, particularly for the $polarization$ and $S12$ features. While intensity prediction ($S11$) generalizes well ($R^2 > 0.95$), polarization features are more sensitive to material-specific interference effects. Future work involves exploring more complex architectures or physics-constrained loss functions to improve cross-composition accuracy.

### Model Architecture Summary (Current):
* **Framework:** Scikit-Learn (`MLPRegressor`) and PyTorch / Skorch
* **Intensity Model (S11):** Hidden Layers: (32, 16, 8), Activation: ReLU, Optimizer: Adam.
* **S12 Model:** Hidden Layers: (32, 16, 8), Activation: Tanh, Optimizer: Adam.
* **Polarization Model (Direct MLP):** Hidden Layers: (32, 16, 8), Activation: ReLU, Optimizer: Adam (Learning Rate: 0.0001).
* **Analytical Polarization Equation Model:** PyTorch deep network `(5 -> 128 -> 128 -> 64 -> 32 -> 4)` with SiLU activations and Sigmoid parameter bounding, trained via Skorch with `LossEquation`.
* **Input Features:** `theta`, `cos_theta`, `sin_theta`, `refractive_index_real`, `log_refractive_index_img` (log-clipped at -9.0), `log_particle_count`, `log_particle_radius`, `size_parameter`
* **Target Features:** `log_S11` (Standard Scaled), `arcsinh_S12` (Standard Scaled), `polarization` (Quantile Transformed), and equation parameters $(b, c_1, c_2, a_0)$.

