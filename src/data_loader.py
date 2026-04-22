import pandas as pd
import pathlib
import os
import re

def parse_dat_file(file_path: pathlib.Path) -> pd.DataFrame:
    """
    Parses a single .dat file to extract metadata and scattering data.
    Supports files with and without MSTM headers.
    """
    # 1. Extract composition from filename
    stem = file_path.stem
    parts = stem.split('_')
    # Filter out empty strings from double underscores and remove last two numeric parts
    name_parts = [p for p in parts[:-2] if p]
    composition = "_".join(name_parts)

    # 2. Check if file has header or is raw data
    with open(file_path, 'r') as f:
        first_line = f.readline()
    
    if first_line.startswith('***'):
        # Standard MSTM output with header
        # Extract number_particles from line 13
        number_particles = None
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 13:
                try:
                    number_particles = int(lines[12].strip())
                except ValueError:
                    print(f"Warning: Could not parse number_particles from line 13 in {file_path}")
        
        # Load data starting from line 52 (skip 51 lines)
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=51, names=['theta', '11', '12', '22', '33', '34', '44'])
    else:
        # Raw data file without header
        # Try to extract number_particles from filename (the last part)
        try:
            number_particles = int(parts[-1])
        except (ValueError, IndexError):
            number_particles = None
            print(f"Warning: Could not determine number_particles for {file_path}")
        
        # Load data from the beginning
        df = pd.read_csv(file_path, sep=r'\s+', names=['theta', '11', '12', '22', '33', '34', '44'])

    # 3. Map columns and filter
    mapping = {
        'theta': 'scattering_angle',
        '11': 'S11',
        '12': 'S12'
    }
    df = df.rename(columns=mapping)
    
    # Keep only requested columns
    df = df[['scattering_angle', 'S11', 'S12']]

    # 4. Remove invalid rows and coerce to numeric
    df = df[(df['scattering_angle'] != 'theta') & (df['scattering_angle'] != 'diffuse')]
    df = df.apply(pd.to_numeric, errors='coerce')

    # 5. Add metadata columns and polarization
    df['polarization'] = - df['S12'] / df['S11']
    df['composition'] = composition
    df['number_particles'] = number_particles
    

    


    return df

def load_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """
    Loads .dat files from the raw directory, processes them, 
    and saves the combined result as a Parquet file.
    """
    raw_path = pathlib.Path(raw_dir)
    processed_path = pathlib.Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    
    # Iterate over all .dat files in the raw directory
    for file_path in raw_path.glob("*.dat"):
        print(f"Processing {file_path.name}...")
        try:
            df = parse_dat_file(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_dfs:
        print("No .dat files found in data/raw.")
        return

    # Consolidate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to Parquet
    output_parquet = processed_path / "dust_data.parquet"
    combined_df.to_parquet(output_parquet, engine="pyarrow")

    # Also save to CSV for easy inspection if needed
    output_csv = processed_path / "dust_data.csv"
    combined_df.to_csv(output_csv, index=False)

    print(f"Data successfully processed and saved to {output_parquet}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Compositions found: {combined_df['composition'].unique()}")

if __name__ == "__main__":
    load_data()
