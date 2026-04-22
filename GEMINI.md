# Project Context: Cosmic Dust ANN
You are an expert assistant in Astrophysics and Machine Learning. 
You are helping me build an Artificial Neural Network (ANN) to predict 
light scattering features from dust composition, particle size, and phase angle.

## Technical Environment
- **Language:** Python 3.10+
- **Key Libraries:** Pandas, NumPy, Scikit-Learn, PyTorch (or TensorFlow)
- **Data Source:** .dat files in `data/raw/`

## Repository Rules
- Follow the established modular structure (`src/`, `data/`, `models/`).
- Prioritize data reproducibility (save scalers in `data/processed/`).
- Use LaTeX for any physics or mathematical formulas.

## Completed Milestones
- **Data Ingestion**: Successfully mapped all `.dat` files from `data/raw/` to a central parquet file. Extracted `composition` from filenames and `number_particles` from file headers. Mapped scattering fields ($theta \rightarrow scattering\_angle$, $11 \rightarrow S11$, $12 \rightarrow S12$).
- **Data Preprocessing**: Implemented `src/preprocessing.py`. Loaded parquet data, included a placeholder for physics-informed processing, split data into train/test sets (80/20), and applied `StandardScaler` to numeric features/targets and `OneHotEncoder` to composition. Scaled datasets saved to `data/features/` and scalers saved to `data/processed/`.

## Current Objective
[PENDING: User to provide current objective details]
