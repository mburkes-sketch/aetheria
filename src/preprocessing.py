import pandas as pd
import numpy as np
import pathlib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(input_file: str = "data/processed/dust_data.parquet", 
                    output_dir: str = "data/features",
                    scaler_dir: str = "data/processed",
                    holdout_composition: str = "OLIVINE M_lab") -> None:
    """
    Loads processed data, applies transformations, splits into train/test/holdout,
    scales numeric features/targets, and saves the results.
    """
    input_path = pathlib.Path(input_file)
    output_path = pathlib.Path(output_dir)
    scaler_path = pathlib.Path(scaler_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    scaler_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_path)

    # --- Section: Physics-Informed Feature Engineering ---
    # These features translate raw physical properties into forms that are
    # more linear or physically meaningful for a Neural Network to learn.

    # 1. Size Parameter (x = 2 * pi * r / lambda)
    # The size parameter is a dimensionless quantity that governs the scattering regime 
    # (Rayleigh vs. Mie). It's more universal than radius alone.
    wavelength = 0.65
    df['size_parameter'] = (2 * np.pi * df['particle_radius']) / wavelength

    # 2. Angular features
    # Neural networks can struggle with raw angles due to periodicity. 
    # cos(theta) and sin_theta) map the angle onto a circle, which often 
    # linearizes the relationship with scattering intensities.
    df['cos_theta'] = np.cos(df['theta'])
    df['sin_theta'] = np.sin(df['theta'])
    df['log_particle_radius'] = np.log10(df['particle_radius'] + 1e-9)
    df['log_particle_count'] = np.log10(df['particle_count'] + 1e-9)
    # Extended lower clip to -9.0 so near-transparent materials (e.g. ice, k≈1.43e-7)
    # map to ≈-6.84 and remain distinguishable from the clip boundary.
    df['log_refractive_index_img'] = np.clip(np.log10(df['refractive_index_img'] + 1e-9), -9.0, None)
    
    df['log_S11'] = np.log10(df['S11'] + 1e-9)
    df['arcsinh_S12'] = np.arcsinh(df['S12'])

    # --- Section: Outlier Removal ---
    # Rationale: Anomalous simulation results or lab errors can introduce extreme 
    # outliers that skew Neural Network gradients. We use the Interquartile Range (IQR) 
    # method applied PER COMPOSITION to preserve material-specific dynamics.
    remove_outliers = True

    if remove_outliers:
      print("Removing outliers based on IQR (targets only)...")
      initial_len = len(df)
      
      target_cols = ['arcsinh_S12', 'polarization']
      mask = pd.Series(True, index=df.index)
      
      for comp in df['composition'].unique():
          comp_mask = df['composition'] == comp
          comp_df = df[comp_mask]
          
          for col in target_cols:
              Q1 = comp_df[col].quantile(0.25)
              Q3 = comp_df[col].quantile(0.75)
              IQR = Q3 - Q1
              # Using a conservative 3.0 multiplier for IQR to only catch extreme outliers
              lower_bound = Q1 - 3.0 * IQR
              upper_bound = Q3 + 3.0 * IQR
              
              # Update the mask for this composition
              mask.loc[comp_mask] &= (df.loc[comp_mask, col] >= lower_bound) & (df.loc[comp_mask, col] <= upper_bound)

      df = df[mask].copy()
      print(f"  Outliers removed: {initial_len - len(df)} (Remaining: {len(df)})")

#-------END OUTLIER REMOVAL------------


    # Separate holdout composition
    if holdout_composition:
        print(f"Separating holdout composition: {holdout_composition}")
        df_holdout = df[df['composition'] == holdout_composition].copy()
        df_rest = df[df['composition'] != holdout_composition].copy()
    else:
        df_holdout = pd.DataFrame()
        df_rest = df

    # Define features and targets
    feature_cols = ['theta', 'cos_theta', 'sin_theta', 
                    'refractive_index_real', 'log_refractive_index_img', 
                    'log_particle_count', 'log_particle_radius', 'size_parameter']
    target_cols = ['log_S11', 'arcsinh_S12', 'polarization']

    # Keep composition for later grouping
    X_rest = df_rest[feature_cols + ['composition']]
    y_rest = df_rest[target_cols]

    # Split the "rest" data into training and testing sets
    print("Splitting remaining data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_rest, y_rest, test_size=0.2, random_state=42
    )

    # Scaling numeric features
    numeric_features = feature_cols
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
        ]
    )

    print("Fitting preprocessor and scaling features...")
    # Scale only the numeric columns, but keep the composition index
    X_train_scaled = preprocessor.fit_transform(X_train[feature_cols])
    X_test_scaled = preprocessor.transform(X_test[feature_cols])
    
    # Convert back to DataFrames and re-attach composition
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_train_df['composition'] = X_train['composition']
    
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    X_test_df['composition'] = X_test['composition']

    if not df_holdout.empty:
        X_holdout_scaled = preprocessor.transform(df_holdout[feature_cols])
        X_holdout_df = pd.DataFrame(X_holdout_scaled, columns=feature_cols, index=df_holdout.index)
        X_holdout_df['composition'] = df_holdout['composition']
        X_holdout_df.to_parquet(output_path / "X_holdout.parquet")

    # --- Section: Specialized Target Scaling ---
    print("Scaling targets...")
    from sklearn.preprocessing import QuantileTransformer
    target_transformer = ColumnTransformer(
        transformers=[
            ('std', StandardScaler(), ['log_S11', 'arcsinh_S12']),
            ('quant', QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42), ['polarization'])
        ]
    )
    
    y_train_scaled = target_transformer.fit_transform(y_train)
    y_test_scaled = target_transformer.transform(y_test)

    if not df_holdout.empty:
        y_holdout_scaled = target_transformer.transform(df_holdout[target_cols])
        y_holdout_df = pd.DataFrame(y_holdout_scaled, columns=target_cols, index=df_holdout.index)
        y_holdout_df.to_parquet(output_path / "y_holdout.parquet")

    y_train_df = pd.DataFrame(y_train_scaled, columns=target_cols, index=y_train.index)
    y_test_df = pd.DataFrame(y_test_scaled, columns=target_cols, index=y_test.index)

    # Save data to parquet
    print(f"Saving features to {output_dir}...")
    X_train_df.to_parquet(output_path / "X_train.parquet")
    X_test_df.to_parquet(output_path / "X_test.parquet")
    y_train_df.to_parquet(output_path / "y_train.parquet")
    y_test_df.to_parquet(output_path / "y_test.parquet")

    # Save scalers/preprocessor for reproducibility
    print(f"Saving scalers to {scaler_dir}...")
    joblib.dump(preprocessor, scaler_path / "feature_preprocessor.joblib")
    joblib.dump(target_transformer, scaler_path / "target_scaler.joblib")

    print("Preprocessing complete.")
    print(f"Training set size: {len(X_train_df)}")
    print(f"Testing set size: {len(X_test_df)}")
    if not df_holdout.empty:
        print(f"Holdout set size ({holdout_composition}): {len(X_holdout_df)}")


def preprocess_data_param_model(input_file: str = "data/processed/dust_data.parquet", 
                    output_dir: str = "data/features",
                    scaler_dir: str = "data/processed",
                    holdout_composition: str = "OLIVINE M_lab") -> None:
  """
  Loads processed data, applies transformations, splits into train/test/holdout,
  scales numeric features/targets, and saves the results.
  """
  input_path = pathlib.Path(input_file)
  output_path = pathlib.Path(output_dir)
  scaler_path = pathlib.Path(scaler_dir)
  
  output_path.mkdir(parents=True, exist_ok=True)
  scaler_path.mkdir(parents=True, exist_ok=True)

  print(f"Loading data from {input_file}...")
  df = pd.read_parquet(input_path)

  # --- Section: Physics-Informed Feature Engineering ---
  # These features translate raw physical properties into forms that are
  # more linear or physically meaningful for a Neural Network to learn.

  # 1. Size Parameter (x = 2 * pi * r / lambda)
  # The size parameter is a dimensionless quantity that governs the scattering regime 
  # (Rayleigh vs. Mie). It's more universal than radius alone.
  wavelength = 0.65
  df['size_parameter'] = (2 * np.pi * df['particle_radius']) / wavelength

  # 2. Angular features
  df['log_particle_radius'] = np.log10(df['particle_radius'] + 1e-9)
  df['log_particle_count'] = np.log10(df['particle_count'] + 1e-9)
  # Extended lower clip to -9.0 so near-transparent materials (e.g. ice, k≈1.43e-7)
  # map to ≈-6.84 and remain distinguishable from the clip boundary.
  df['log_refractive_index_img'] = np.clip(np.log10(df['refractive_index_img'] + 1e-9), -9.0, None)
  
  df['log_S11'] = np.log10(df['S11'] + 1e-9)

  # --- Section: Outlier Removal ---
  # Rationale: Anomalous simulation results or lab errors can introduce extreme 
  # outliers that skew Neural Network gradients. We use the Interquartile Range (IQR) 
  # method applied PER COMPOSITION to preserve material-specific dynamics.
  remove_outliers = True

  if remove_outliers:
    print("Removing outliers based on IQR (targets only)...")
    initial_len = len(df)
    
    target_cols = ['polarization']
    mask = pd.Series(True, index=df.index)
    
    for comp in df['composition'].unique():
        comp_mask = df['composition'] == comp
        comp_df = df[comp_mask]
        
        for col in target_cols:
            Q1 = comp_df[col].quantile(0.25)
            Q3 = comp_df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Using a conservative 3.0 multiplier for IQR to only catch extreme outliers
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            # Update the mask for this composition
            mask.loc[comp_mask] &= (df.loc[comp_mask, col] >= lower_bound) & (df.loc[comp_mask, col] <= upper_bound)

    df = df[mask].copy()
    print(f"  Outliers removed: {initial_len - len(df)} (Remaining: {len(df)})")

#-------END OUTLIER REMOVAL------------

  # Separate holdout composition
  if holdout_composition:
      print(f"Separating holdout composition: {holdout_composition}")
      df_holdout = df[df['composition'] == holdout_composition].copy()
      df_rest = df[df['composition'] != holdout_composition].copy()
  else:
      df_holdout = pd.DataFrame()
      df_rest = df

  # Define features and targets
  feature_cols = ['refractive_index_real', 'log_refractive_index_img', 
                  'log_particle_count', 'log_particle_radius', 'size_parameter']
  target_cols = ['theta', 'log_S11', 'polarization']

  # Keep composition for later grouping
  X_rest = df_rest[feature_cols + ['composition']]
  y_rest = df_rest[target_cols]

  # Split the "rest" data into training and testing sets
  print("Splitting remaining data into training and testing sets (80/20)...")
  X_train, X_test, y_train, y_test = train_test_split(
      X_rest, y_rest, test_size=0.2, random_state=42
  )

  # Scaling numeric features
  numeric_features = feature_cols
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numeric_features),
      ]
  )

  print("Fitting preprocessor and scaling features...")
  # Scale only the numeric columns, but keep the composition index
  X_train_scaled = preprocessor.fit_transform(X_train[feature_cols])
  X_test_scaled = preprocessor.transform(X_test[feature_cols])
  
  # Convert back to DataFrames and re-attach composition
  X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
  X_train_df['composition'] = X_train['composition']
  
  X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
  X_test_df['composition'] = X_test['composition']

  if not df_holdout.empty:
      X_holdout_scaled = preprocessor.transform(df_holdout[feature_cols])
      X_holdout_df = pd.DataFrame(X_holdout_scaled, columns=feature_cols, index=df_holdout.index)
      X_holdout_df['composition'] = df_holdout['composition']
  
  if not df_holdout.empty:
      y_holdout_df = pd.DataFrame(df_holdout[target_cols], columns=target_cols, index=df_holdout.index)


  y_train_df = pd.DataFrame(y_train, columns=target_cols, index=y_train.index)
  y_test_df = pd.DataFrame(y_test, columns=target_cols, index=y_test.index)

  train_df = pd.concat([X_train_df, y_train_df], axis=1)
  test_df = pd.concat([X_test_df, y_test_df], axis=1)

  collapsed_train_df = train_df.groupby(feature_cols).agg({
      'theta': list,
      'log_S11': list,
      'polarization': list,
  }).reset_index()

  collapsed_test_df = test_df.groupby(feature_cols).agg({
      'theta': list,
      'log_S11': list,
      'polarization': list,
  }).reset_index()

  print(collapsed_train_df)

  X_train_df = collapsed_train_df[feature_cols]
  X_test_df = collapsed_test_df[feature_cols]

  y_train_df = collapsed_train_df[target_cols]
  y_test_df = collapsed_test_df[target_cols]

  if not df_holdout.empty:
    holdout_df = pd.concat([X_holdout_df, y_holdout_df], axis=1)
    df_holdout_f = holdout_df.groupby(feature_cols).agg({
        'theta': list,
        'log_S11': list,
        'polarization': list,
    }).reset_index()

    X_holdout_df_f = df_holdout_f[feature_cols]
    y_holdout_df_f = df_holdout_f[target_cols]

    X_holdout_df_f.to_parquet(output_path / "X_holdout_f.parquet")
    y_holdout_df_f.to_parquet(output_path / "y_holdout_f.parquet")

  # Save data to parquet
  print(f"Saving features to {output_dir}...")
  X_train_df.to_parquet(output_path / "X_train_f.parquet")
  X_test_df.to_parquet(output_path / "X_test_f.parquet")
  y_train_df.to_parquet(output_path / "y_train_f.parquet")
  y_test_df.to_parquet(output_path / "y_test_f.parquet")

  # Save scalers/preprocessor for reproducibility
  print(f"Saving scalers to {scaler_dir}...")
  joblib.dump(preprocessor, scaler_path / "feature_preprocessor_f.joblib")


  print("Preprocessing complete.")
  print(f"Training set size: {len(X_train_df)}")
  print(f"Testing set size: {len(X_test_df)}")
  if not df_holdout.empty:
      print(f"Holdout set size ({holdout_composition}): {len(X_holdout_df)}")

  return


if __name__ == "__main__":
    holdout_composition = "OLIVINE M_lab"
    preprocess_data(holdout_composition=holdout_composition)
    preprocess_data_param_model(holdout_composition=holdout_composition)
