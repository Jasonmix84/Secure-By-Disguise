import pandas as pd
import argparse
import os

def create_recovered_csv(base_csv_path, encrypted_dir, recovered_dir, output_csv_path):
    """
    Create a CSV for the recovered dataset by transforming paths from the training CSV.
    
    Args:
        base_csv_path: Path to original training CSV
        encrypted_dir: Original encrypted directory path (what to replace)
        recovered_dir: Recovered directory path (what to replace with)
        output_csv_path: Path to save the new CSV
    """
    df = pd.read_csv(base_csv_path)
    
    # Transform paths
    df['path'] = df['path'].str.replace(encrypted_dir, recovered_dir)
    
    # Verify paths exist
    missing_count = 0
    for idx, path in df['path'].items():
        if not os.path.exists(path):
            print(f"WARNING: Path does not exist: {path}")
            missing_count += 1
    
    if missing_count > 0:
        print(f"\nWarning: {missing_count} out of {len(df)} paths do not exist!")
    else:
        print(f"All {len(df)} paths verified!")
    
    # Save new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nRecovered CSV saved to: {output_csv_path}")
    print(f"Total samples: {len(df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_csv', type=str, required=True, help='Path to original training CSV')
    parser.add_argument('--encrypted_dir', type=str, required=True, help='Encrypted directory to replace (e.g., "ModelData/Breast/RMT-B4N4")')
    parser.add_argument('--recovered_dir', type=str, required=True, help='Recovered directory to replace with (e.g., "Data/Breast/recovered/recovered-RMT-B4N4-kp1")')
    parser.add_argument('--output_csv', type=str, required=True, help='Output path for recovered CSV')
    
    args = parser.parse_args()
    
    create_recovered_csv(args.base_csv, args.encrypted_dir, args.recovered_dir, args.output_csv)


if __name__ == '__main__':
    main()
