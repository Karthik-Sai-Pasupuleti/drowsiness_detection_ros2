import h5py
import pandas as pd
import os

def convert_h5_to_csv(h5_file_path):
    if not os.path.exists(h5_file_path):
        print(f"Error: File {h5_file_path} not found.")
        return

    summary_rows = []
    
    try:
        with h5py.File(h5_file_path, "r") as f:
            # Sort windows numerically (window_0, window_1...)
            window_keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
            
            for window_key in window_keys:
                group = f[window_key]
                row = {"window_name": window_key}
                
                # Extract all attributes (Metrics + Labels)
                for attr_name, attr_value in group.attrs.items():
                    row[attr_name] = attr_value
                
                summary_rows.append(row)

        if not summary_rows:
            print("No data found in HDF5 file.")
            return

        df_summary = pd.DataFrame(summary_rows)
        
        # Define output path
        summary_csv = h5_file_path.replace(".h5", "_summary.csv")
        
        # Check for permission before saving
        try:
            df_summary.to_csv(summary_csv, index=False)
            print(f"✅ Success! Summary saved to: {summary_csv}")
        except PermissionError:
            print(f"❌ PERMISSION DENIED: Could not write to {summary_csv}.")
            print("Is the CSV file open in Excel or another program?")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure this path is correct for your system
    path_to_h5 = "/home/karthik/Desktop/ws/drowsiness_detection_ros2/drowsiness_data/karthik_pilot_test/session_data.h5"
    convert_h5_to_csv(path_to_h5)