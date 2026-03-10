#!/usr/bin/env python3

import h5py
import numpy as np
import pandas as pd
import json
import os
import argparse

def decode_val(val):
    """Decode bytes or numpy scalars to native Python types."""
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val

def array_to_json(arr):
    """Serialize a numpy array to a JSON string for CSV storage."""
    if isinstance(arr, h5py.Dataset):
        arr = arr[()]
    if isinstance(arr, np.ndarray):
        return json.dumps(arr.tolist())
    return str(arr)

def convert_session_h5_to_csv(h5_path, output_csv=None):
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found.")
        return

    if output_csv is None:
        output_csv = h5_path.replace(".h5", "_complete_export.csv")

    print(f"Reading HDF5: {h5_path}")
    
    rows = []
    
    with h5py.File(h5_path, "r") as f:
        # Sort windows numerically by ID
        win_keys = []
        for k in f.keys():
            if k.startswith("window_"):
                try:
                    win_keys.append((int(k.split("_")[1]), k))
                except:
                    pass
        
        win_keys.sort()
        
        for _, win_name in win_keys:
            win_group = f[win_name]
            row = {"window_name": win_name}
            
            # 1. Extract all attributes (Metrics + Multiple Annotator Labels)
            for attr_name, attr_value in win_group.attrs.items():
                row[attr_name] = decode_val(attr_value)
                
            # 2. Extract Raw datasets
            # Check for 'raw_data' group (standard structure)
            raw_group = None
            if "raw_data" in win_group:
                raw_group = win_group["raw_data"]
            elif "rawdata" in win_group: # backup naming
                raw_group = win_group["rawdata"]
            
            if raw_group:
                for ds_name in raw_group.keys():
                    ds = raw_group[ds_name]
                    if isinstance(ds, h5py.Dataset):
                        # We store raw data as a JSON string in the CSV cell
                        # This preserves the full sequence for that window
                        row[f"raw_{ds_name}"] = array_to_json(ds)
            
            rows.append(row)

    if not rows:
        print("No window data found in HDF5.")
        return

    df = pd.DataFrame(rows)
    
    # Reorder columns: window_name, window_id (if exists) first, then metrics/labels, then raw data
    cols = list(df.columns)
    first_cols = ["window_name", "window_id"]
    priority_cols = [c for c in first_cols if c in cols]
    raw_cols = [c for c in cols if c.startswith("raw_")]
    other_cols = [c for c in cols if c not in priority_cols and c not in raw_cols]
    
    final_cols = priority_cols + other_cols + raw_cols
    df = df[final_cols]
    
    try:
        df.to_csv(output_csv, index=False)
        print(f"Done! Conversion complete.")
        print(f"Exported {len(df)} windows with {len(df.columns)} columns.")
        print(f"Saved to: {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Drowsiness session HDF5 to a complete CSV export.")
    parser.add_argument("input", help="Path to the .h5 file", nargs='?', 
                        default="/home/karthik/Desktop/ws/drowsiness_detection_ros2/drowsiness_data/Anthony/session_data.h5")
    parser.add_argument("--output", help="Optional output CSV path")
    
    args = parser.parse_args()
    convert_session_h5_to_csv(args.input, args.output)
