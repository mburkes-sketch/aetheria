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
    - **Calculations**: Calculated complex dielectric constants ($\epsilon = \epsilon' + i\epsilon''$) and normalized $S_{11}$ at $60^\circ$ for cross-format consistency.
    - **Data Fidelity Visualization**: Added automatic generation of diagnostic plots for each composition (S11, S12, polarization) in `data/processed/graphs/` to verify raw data quality before preprocessing.
- **Data Preprocessing & Feature Engineering**: Implemented `src/preprocessing.py`.
    - **Physics-Informed Features**: Added `size_parameter` ($x = \frac{2\pi r}{\lambda}$), `refractive_index` ($m = \sqrt{\epsilon}$), and trigonometric transforms for $\theta$ ($\cos\theta, \sin\theta$).
    - **Scaling Strategy**: Applied `StandardScaler` to most features and `QuantileTransformer` to the $polarization$ target to handle its non-Gaussian distribution.
    - **Data Splitting**: 80/20 train/test split.
- **Model Training**: Implemented separate Scikit-Learn `MLPRegressor` models to capture the different dynamics of light intensity and polarization:
    - **Intensity Model**: A 3-layer MLP (128, 64, 32 units) predicting $\log(S11)$ and $\log(S12)$.
    - **Polarization Model**: A deeper 3-layer MLP (512, 256, 128 units) specifically tuned for the subtle $polarization$ feature.
- **Evaluation & Visualization**: Developed `src/evaluate.py` to calculate $R^2$ metrics on scaled and original data. 
    - **Visualization Suite**: Added tools to automatically generate and save comparison plots for predicted vs. ground-truth curves for $S11$ (intensity), $S12$, and $polarization$ across all compositions in the `results/` directory.
    - **Generalizability Testing**: Implemented a "Holdout Composition" strategy in `src/preprocessing.py`. This allows training on a subset of materials and validating performance on a completely unseen composition (e.g., `halley_dust`), providing a rigorous measure of the model's physical generalizability.

## Current Objective
The primary focus is to bridge the performance gap between seen and unseen compositions, particularly for the $polarization$ and $S12$ features. While intensity prediction ($S11$) generalizes well ($R^2 > 0.95$), polarization features are more sensitive to material-specific interference effects. Future work involves exploring more complex architectures or physics-constrained loss functions to improve cross-composition accuracy.

### Model Architecture Summary (Current):
* **Framework:** Scikit-Learn (`MLPRegressor`)
* **Intensity Model:** Hidden Layers: (128, 64, 32), Activation: ReLU, Optimizer: Adam.
* **Polarization Model:** Hidden Layers: (512, 256, 128), Activation: ReLU, Optimizer: Adam (Learning Rate: 0.0001).
* **Target Features:** `log_S11`, `log_S12`, `polarization`.

