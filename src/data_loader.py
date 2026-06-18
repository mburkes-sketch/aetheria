import pandas as pd
import pathlib
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use a non-interactive backend for plotting in CLI
matplotlib.use('Agg')

# Material mapping to complex refractive index (n, k)
# n = real part (scattering), k = imaginary part (absorption)
MAT_TO_REF_IND = {
    "halley_dust": (1.98, 0.48),
    "am_silicate": (1.677, 0.0044),
    "amorphous_silicate": (1.677, 0.0044),
    "water_ice": (1.308, 0.000000143),
    "water__ice": (1.308, 0.000000143),
    "ice": (1.308, 0.000000143),
    "co2_ice": (1.41, 0.00000105),
    "pyroxene": (1.702, 0.0212),
    "iron_pyroxene": (1.702, 0.0212),
    "iron_rich_pyroxene": (1.702, 0.0212),
    "olivine": (1.743, 0.066),
    "am_carbon": (2.14, 0.805),
    "amorphous_carbon": (2.14, 0.805),
    "amorphous__carbon": (2.14, 0.805),
    "pyrrhotite": (1.7, 1.86),
    "ir_olivine": (1.85, 0.00077),
    "iron": (1.85, 0.00077), 
    "tholin_ice": (1.54, 0.0012),
    "titan_tholins": (1.557, 0.0009)
}

def parse_h5_file(file_path: pathlib.Path) -> pd.DataFrame:
    """
    Parses a single .h5 file to extract Mueller matrix data and associated metadata.
    Rationale: .h5 files contain the raw scattering matrices and metadata encoded in their filenames.
    """
    # 1. Extract metadata (composition, size, count) from the filename string
    stem = file_path.stem
    parts = stem.split('_')
    
    # Filter out empty strings from double underscores and remove the last three suffix parts
    useful_parts = [p for p in parts[:-3] if p]
    
    # Logic to detect if size or particle count comes first in the naming convention
    if useful_parts[-1][-4:] == 'size':
        # Example: "01size" -> 0.1 microns
        size = int(useful_parts[-1][:-4]) / 10
        # Example: "8k" -> 8000 particles
        particle_count = int(useful_parts[-2][:-1]) * 1000
    else:
        size = int(useful_parts[-2][:-4]) / 10
        particle_count = int(useful_parts[-1][:-1]) * 1000
    
    # Everything before the size/count is treated as the material composition
    composition = "_".join(useful_parts[:-2])

    df = pd.DataFrame()

    # Open the HDF5 file in read-only mode
    with h5py.File(file_path, 'r') as f:
        # Extract the dataset (usually the second key in these files)
        mueller_key = list(f.keys())[1]
        data = f[mueller_key]

        # Convert HDF5 dataset to a DataFrame and transpose
        # Original shape is (variables, angles) -> we want (angles, variables)
        df_data = pd.DataFrame(list(data)).transpose()
        df_data = df_data.apply(pd.to_numeric, errors='coerce')


        pd.set_option('display.max_rows', None)

        # Map standard scattering columns
        df['theta'] = df_data.iloc[:, 0]      # Scattering angle in radians
        df['S11'] = df_data.iloc[:, 1]        # Intensity
        df['S12'] = df_data.iloc[:, 2]        # Cross-polarization element
        
        # 2. Data Cleaning: Filter out extremely small S11 values
        # Rationale: Polarization = -S12/S11. If S11 is near zero, 
        # small numerical errors in S12 blow up into massive polarization outliers.
        # print(df[['theta', 'S11', 'S12']])
        df = df[df['S11'] > 1e-4].copy()

        # df['S12'] = df['S12']/10
        
        # df['S12'] = df['S12']/abs(df['S12'][60]) 

        # 3. Calculate Polarization
        df['polarization'] = - df['S12'] / df['S11']

        # Divide by S11(deg30) to align with lab data
        normalization_factor = df['S11'][60]
        df['S12'] = df['S12']/normalization_factor
        df['S11'] = df['S11']/normalization_factor #Makes predicting lab s11 better, but not polarization
        
        # Safety check: Replace any resulting Infs or NaNs with 0
        df['polarization'] = df['polarization'].fillna(0).replace([np.inf, -np.inf], 0)

        # Attach metadata to every row in the dataframe
        df['refractive_index_real'] = MAT_TO_REF_IND[composition][0]
        df['refractive_index_img'] = MAT_TO_REF_IND[composition][1]
        df['particle_radius'] = size
        df['particle_count'] = particle_count
        df['composition'] = composition

    return df

def parse_dat_file(file_path: pathlib.Path) -> pd.DataFrame:
    """
    Parses an MSTM .dat file, handling both headered and headless variants.
    """
    stem = file_path.stem
    # Handle double underscores and extract parts
    parts = [p for p in stem.split('_') if p]
    
    # Last part is particle count per user instruction
    try:
        particle_count = int(parts[-1])
    except (ValueError, IndexError):
        particle_count = 10000
        
    # Radius is assumed to be 0.1
    size = 0.1
    
    # Composition is everything before the last two numeric/identifying parts (e.g., '64' and '16384')
    composition_parts = parts[:-2]
    composition = "_".join(composition_parts)
    
    # Normalize composition names to match .h5 conventions
    norm_comp = {
        "amorphous_carbon": "am_carbon",
        "amorphous_silicate": "am_silicate",
        "water_ice": "water_ice", # already matches but for safety
        "water__ice": "water_ice"
    }
    composition = norm_comp.get(composition, composition)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Try to extract refractive index from header
    content = "".join(lines[:100])
    ri_match = re.search(r"ref index scale factors\s*\n\s*[\d\.\-Ee]+\s+([\d\.\-Ee]+)\s+([\d\.\-Ee]+)", content)
    if ri_match:
        n = float(ri_match.group(1))
        k = float(ri_match.group(2))
    else:
        # Fallback to mapping
        n, k = MAT_TO_REF_IND.get(composition, (1.6, 0.001))

    # Locate the start of the data table
    start_idx = 0
    header_found = False
    for i, line in enumerate(lines):
        if 'theta' in line.lower() and '11' in line:
            start_idx = i + 1
            header_found = True
            break
    
    if not header_found:
        # Check if it's headless (starts with numbers)
        for i, line in enumerate(lines):
            if re.match(r"^\s*[\d\.\-Ee]+", line):
                start_idx = i
                break
                
    # Read the numerical data columns
    data = []
    # We only want the first 181 rows of data if multiple blocks exist (like 'diffuse scattering matrix')
    count = 0
    for line in lines[start_idx:]:
        parts_list = line.split()
        if len(parts_list) >= 3:
            try:
                # theta, 11, 12, ...
                data.append([float(p) for p in parts_list[:3]])
                count += 1
                if count >= 181: # MSTM usually has 181 angles
                    break
            except ValueError:
                if count > 0: # Stop if we hit non-numeric line after starting to read data
                    break
                continue
                
    if not data:
        raise ValueError(f"No numeric data found in {file_path}")

    df_raw = pd.DataFrame(data, columns=['theta', 'S11', 'S12/S11'])

    df = pd.DataFrame()
    # Convert angle (degrees) to theta (radians)
    df['theta'] = df_raw['theta'] * np.pi / 180
    df['S11'] = df_raw['S11']

    # In MSTM, col3 is -S12/S11 (Linear polarization ratio)
    # Physical S12 = (col3) * S11
    df['S12'] = df_raw['S12/S11'] * df_raw['S11']

    # Polarization = -S12/S11. In MSTM, col3 IS the polarization ratio (sometimes with a sign flip)
    # Rationale: Polarization = -S12/S11. If col3 is S12/S11, then P = -col3.
    df['polarization'] = - df_raw['S12/S11']

    # Data Cleaning (identical to parse_h5_file)
    df = df[df['S11'] > 1e-4].copy()

    # Normalize S11 at 60 degrees to match synthetic .h5 conventions
    try:
        # Finding the row closest to 60 degrees
        idx_60 = (df['theta'] - 60 * np.pi / 180).abs().idxmin()
        norm_val = df.loc[idx_60, 'S11']
        if norm_val != 0:
            # Re-scale both S11 and S12 by the same factor to preserve the ratio (polarization)
            df['S11'] = df['S11'] / norm_val
            df['S12'] = df['S12'] / norm_val
    except Exception:
        pass

    # Safety check: Replace any resulting Infs or NaNs with 0
    df['polarization'] = df['polarization'].fillna(0).replace([np.inf, -np.inf], 0)
    # Attach metadata
    df['refractive_index_real'] = n
    df['refractive_index_img'] = k
    df['particle_radius'] = size
    df['particle_count'] = particle_count
    df['composition'] = composition

    return df

def generate_data_plots(df: pd.DataFrame, output_dir: pathlib.Path):
    """
    Generates diagnostic plots for S11, S12, and polarization for each composition.
    Rationale: Visual verification of data fidelity before preprocessing.
    """
    graphs_path = output_dir / "graphs"
    (graphs_path / "S11").mkdir(parents=True, exist_ok=True)
    (graphs_path / "S12").mkdir(parents=True, exist_ok=True)
    (graphs_path / "polarization").mkdir(parents=True, exist_ok=True)
    
    unique_compositions = df['composition'].unique()
    
    for comp in unique_compositions:
        comp_df = df[df['composition'] == comp]
        safe_name = str(comp).replace(" ", "_").lower()
        
        # 1. S11 Plot
        plt.figure(figsize=(10, 6))
        for (radius, count), group in comp_df.groupby(['particle_radius', 'particle_count']):
            plt.scatter(np.degrees(group['theta']), group['S11'], label=f"r={radius}, N={count}", s=10, alpha=0.5)
        plt.yscale('log')
        plt.title(f"Raw S11 Intensity: {comp}")
        plt.xlabel("Angle (deg)")
        plt.ylabel("S11 (Normalized @ 60 deg)")
        if len(comp_df.groupby(['particle_radius', 'particle_count'])) < 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_path / "S11" / f"{safe_name}_s11.png", dpi=150)
        plt.close()

        # 2. S12 Plot
        plt.figure(figsize=(10, 6))
        for (radius, count), group in comp_df.groupby(['particle_radius', 'particle_count']):
            plt.scatter(np.degrees(group['theta']), group['S12'], label=f"r={radius}, N={count}", s=10, alpha=0.5)
        plt.title(f"Raw S12 Element: {comp}")
        plt.xlabel("Angle (deg)")
        plt.ylabel("S12")
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        if len(comp_df.groupby(['particle_radius', 'particle_count'])) < 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_path / "S12" / f"{safe_name}_s12.png", dpi=150)
        plt.close()

        # 3. Polarization Plot
        plt.figure(figsize=(10, 6))
        for (radius, count), group in comp_df.groupby(['particle_radius', 'particle_count']):
            plt.scatter(np.degrees(group['theta']), group['polarization'], label=f"r={radius}, N={count}", s=10, alpha=0.5)
        plt.title(f"Raw Polarization: {comp}")
        plt.xlabel("Angle (deg)")
        plt.ylabel("Polarization (-S12/S11)")
        # Remove fixed ylim to allow dynamic scaling for low-polarization materials
        if len(comp_df.groupby(['particle_radius', 'particle_count'])) < 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphs_path / "polarization" / f"{safe_name}_polarization.png", dpi=150)
        plt.close()

def parse_lab_file(file_path: pathlib.Path) -> pd.DataFrame:
    """
    Parses an experimental lab data text file (e.g., from the Amsterdam-Granada Database).
    Rationale: Lab data has a specific header format and space-delimited columns.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 1. Extract metadata from header using regex
    # Composition: First line before " - "
    composition = lines[0].split(' - ')[0].strip()
    
    # Refractive index: "refractive index: 1.62 + i0.00001"
    ri_match = re.search(r"refractive index:\s*([\d\.]+)\s*\+\s*i([\d\.]+)", "".join(lines[:10]))
    n = float(ri_match.group(1)) if ri_match else 1.6
    k = float(ri_match.group(2)) if ri_match else 0.001
    
    # Radius: "radius: 0.1"
    radius_match = re.search(r"radius:\s*([\d\.]+)", "".join(lines[:10]))
    radius = float(radius_match.group(1)) if radius_match else 0.1
    
    # Particle count: "particle count: 16000"
    count_match = re.search(r"particle count:\s*([\d\.]+)", "".join(lines[:10]))
    particle_count = float(count_match.group(1)) if count_match else 10000

    # 2. Locate the start of the data table
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('angle'):
            start_idx = i + 1
            break
            
    # Read the numerical data columns
    data = []
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                # angle, F11, error, -F12/F11, ...
                data.append([float(p) for p in parts])
            except ValueError:
                continue
                
    df_raw = pd.DataFrame(data)
    
    df = pd.DataFrame()
    # Convert angle (degrees) to theta (radians)
    df['theta'] = df_raw.iloc[:, 0] * np.pi / 180
    # S11 (normalized at 30 deg in these files)
    df['S11'] = df_raw.iloc[:, 1]
    # Polarization (-F12/F11)
    df['polarization'] = df_raw.iloc[:, 3]
    # Reconstruct S12 element: polarization = -S12/S11 => S12 = -polarization * S11
    df['S12'] = - df['polarization'] * df['S11']
    
    # Fill in metadata
    df['refractive_index_real'] = n
    df['refractive_index_img'] = k
    df['particle_radius'] = radius
    df['particle_count'] = particle_count
    df['composition'] = composition + "_lab" # Suffix to distinguish from synthetic data

    return df

def load_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """
    Aggregates synthetic (.h5), MSTM (.dat), and lab (.txt) data into a single processed parquet file.
    """
    raw_path = pathlib.Path(raw_dir)
    lab_path = raw_path / "lab_data"
    mstm_path = raw_path / "MSTM_data"
    fastmm_path = raw_path / "FASTMM_data"
    processed_path = pathlib.Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    # 1. Process synthetic (.h5) data
    print("Searching for synthetic (.h5) data...")
    h5_search_path = fastmm_path if fastmm_path.exists() else raw_path
    for file_path in h5_search_path.glob("*.h5"):
        print(f"  Processing {file_path.name}...")
        try:
            df = parse_h5_file(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    # 2. Process MSTM (.dat) data
    print("\nSearching for MSTM (.dat) data...")
    if mstm_path.exists():
        for file_path in mstm_path.glob("*.dat"):
            print(f"  Processing MSTM file: {file_path.name}...")
            try:
                df = parse_dat_file(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
    else:
        print(f"  MSTM data directory not found: {mstm_path}")

    # 3. Process experimental lab (.txt) data
    print("\nSearching for lab (.txt) data...")
    if lab_path.exists():
        for file_path in lab_path.glob("*.txt"):
            print(f"  Processing lab file: {file_path.name}...")
            try:
                df = parse_lab_file(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
    else:
        print(f"  Lab data directory not found: {lab_path}")

    if not all_dfs:
        print("\nNo data files found.")
        return

    # Consolidate everything into one large dataset
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to Parquet
    output_parquet = processed_path / "dust_data.parquet"
    combined_df.to_parquet(output_parquet, engine="pyarrow")
    
    # Save to CSV for inspection
    output_csv = processed_path / "dust_data.csv"
    combined_df.to_csv(output_csv, index=False)

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots in data/processed/graphs...")
    generate_data_plots(combined_df, processed_path)

    print(f"\nTotal rows loaded: {len(combined_df)}")
    print(f"Compositions found: {combined_df['composition'].unique()}")
    print(f"Data saved to {output_parquet}")

if __name__ == "__main__":
    load_data()
