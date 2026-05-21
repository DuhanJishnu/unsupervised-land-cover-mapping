import scipy.io as sio
import pandas as pd
import numpy as np
import os

def mat_to_csv(mat_file, output_csv=None):
    """
    Convert a .mat file to CSV format
    
    Args:
        mat_file (str): Path to the .mat file
        output_csv (str): Path to output CSV file. If None, uses mat_file name with .csv extension
    """
    
    # Check if file exists
    if not os.path.exists(mat_file):
        print(f"Error: File '{mat_file}' not found!")
        return False
    
    # Set output filename if not provided
    if output_csv is None:
        output_csv = mat_file.replace('.mat', '.csv')
    
    try:
        # Load the .mat file
        print(f"Loading {mat_file}...")
        mat_data = sio.loadmat(mat_file)
        
        # Display available variables in the .mat file
        print(f"\nVariables in the .mat file:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"  - {key}: shape {mat_data[key].shape}")
        
        # Find the main data variable (usually not starting with __)
        data_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                data_key = key
                break
        
        if data_key is None:
            print("Error: No data variable found in .mat file!")
            return False
        
        data = mat_data[data_key]
        print(f"\nUsing variable '{data_key}' for conversion")
        print(f"Original shape: {data.shape}")
        
        # Reshape data if it's 3D (hyperspectral image)
        if len(data.shape) == 3:
            height, width, bands = data.shape
            print(f"Data is 3D: {height}x{width}x{bands}")
            # Reshape to 2D: each row is a pixel, each column is a band
            data_2d = data.reshape(height * width, bands)
            print(f"Reshaped to 2D: {data_2d.shape}")
        else:
            print(f"Data is {len(data.shape)}D, using as is")
            data_2d = data.flatten().reshape(-1, 1) if len(data.shape) == 1 else data
        
        # Convert to DataFrame and save as CSV
        print(f"\nConverting to CSV and saving to {output_csv}...")
        df = pd.DataFrame(data_2d)
        df.to_csv(output_csv, index=False, header=False)
        
        print(f"✓ Successfully converted to CSV!")
        print(f"  Output file: {output_csv}")
        print(f"  Data shape: {df.shape}")
        
        return True
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert .mat to CSV
    mat_files = ["Indian_pines_corrected", "Indian_pines_gt", "PaviaU_gt", "PaviaU"]
    for filename in mat_files:
        mat_file = os.path.join(current_dir, "mat_files", f"{filename}.mat")
        csv_file = os.path.join(current_dir, "csv_files", f"{filename}.csv")
        mat_to_csv(mat_file, csv_file)
