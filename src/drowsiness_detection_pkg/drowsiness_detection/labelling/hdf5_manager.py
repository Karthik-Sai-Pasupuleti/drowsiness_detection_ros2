#!/usr/bin/env python3

"""
This module contains functions to save and export drowsiness detection data
to/from HDF5 files and CSV format. It supports multiple annotators and
handles raw data, metrics, images, and labels
"""
import os
import time
import json
import h5py
import numpy as np
import pandas as pd
import cv2

# --- Paths ---
DRIVER_ID = "driver_1"
HDF5_FILE = f"drowsiness_data/{DRIVER_ID}_session.h5"
CSV_FILE = f"drowsiness_data/{DRIVER_ID}_session.csv"
IMAGES_FOLDER = f"drowsiness_data/images/{DRIVER_ID}"
os.makedirs(IMAGES_FOLDER, exist_ok=True)


# ------------------- HDF5 Save -------------------
def save_to_hdf5(window_id, window_data, annotator, labels):
    """
    Save window data, metrics, images, and labels to HDF5.
    Supports multiple annotators and flattened labels including:
    - action_1 ... action_N
    - voice_feedback
    """
    os.makedirs(os.path.dirname(HDF5_FILE), exist_ok=True)
    try:
        with h5py.File(HDF5_FILE, "a") as hf:
            window_group = hf.require_group(f"window_{window_id}")

            # --- Raw data ---
            raw_data_group = window_group.require_group("raw_data")
            for key, array_data in window_data.get("raw_data", {}).items():
                if array_data is not None and len(array_data) > 0:
                    dset = np.array(array_data, dtype=np.float32)
                    if key in raw_data_group:
                        del raw_data_group[key]
                    raw_data_group.create_dataset(
                        key, data=dset, compression="gzip", compression_opts=9
                    )

            # --- Metrics ---
            metrics_group = window_group.require_group("metrics")
            for key, value in window_data.get("metrics", {}).items():
                if key in metrics_group:
                    del metrics_group[key]
                if isinstance(value, str):
                    value = np.string_(value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    value = np.array(value)
                metrics_group.create_dataset(key, data=value)

            # --- Images ---
            images = window_data.get("images", [])
            if images:
                images_group = window_group.require_group("images")
                for i, img_bytes in enumerate(images):
                    if f"img_{i}" in images_group:
                        del images_group[f"img_{i}"]
                    # Store JPEG bytes as string
                    images_group.create_dataset(f"img_{i}", data=np.void(img_bytes))

            # --- Annotations ---
            annotations_group = window_group.require_group("annotations")
            annotator_group = annotations_group.require_group(annotator)

            # Flatten all label keys including action_1…action_5 and voice_feedback
            for key, value in labels.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                annotator_group.attrs.modify(key, value)

        print(
            f"[HDF5] Successfully saved window {window_id} for annotator '{annotator}'"
        )
    except Exception as e:
        print(f"[HDF5 ERROR] Failed to save window {window_id}: {e}")
        raise IOError(f"Failed to save data to HDF5: {e}")


# ------------------- CSV Export -------------------
def hdf5_to_csv_export():
    data_rows = []

    try:
        with h5py.File(HDF5_FILE, "r") as hf:
            for window_name in hf.keys():
                window_group = hf[window_name]
                # --- Metrics ---
                metrics = {}
                if "metrics" in window_group:
                    for k, v in window_group["metrics"].items():
                        val = v[()]
                        if hasattr(val, "item"):
                            val = val.item()
                        metrics[k] = val

                raw_data = {}
                if "raw_data" in window_group:
                    for k, v in window_group["raw_data"].items():
                        raw_data[k] = ", ".join(map(str, v[()]))

                # --- Images ---
                image_filenames = []
                if "images" in window_group:
                    for i, img_key in enumerate(sorted(window_group["images"].keys())):
                        img_bytes = window_group["images"][img_key][()].tobytes()
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        img_filename = f"{window_name}_frame_{i}.jpg"
                        img_path = os.path.join(IMAGES_FOLDER, img_filename)
                        os.makedirs(IMAGES_FOLDER, exist_ok=True)
                        cv2.imwrite(img_path, img)
                        image_filenames.append(img_filename)

                combined_labels = {}
                if "annotations" in window_group:
                    annotations_group = window_group["annotations"]
                    for annotator in annotations_group:
                        annotator_group = annotations_group[annotator]
                        for k, v in annotator_group.attrs.items():
                            if isinstance(v, bytes):
                                v = v.decode("utf-8")
                            key = f"{annotator}_{k}"
                            combined_labels[key] = v

                # --- Build row ---
                row = {
                    "window_id": window_name.replace("window_", ""),
                    "images": ", ".join(image_filenames),
                    **metrics,
                    **raw_data,
                    **combined_labels,
                }
                data_rows.append(row)

    except FileNotFoundError:
        print(f"[HDF5] File not found at {HDF5_FILE}")
        return
    except Exception as e:
        print(f"[HDF5 ERROR] CSV export failed: {e}")
        return

    # --- Save CSV ---
    if data_rows:
        df = pd.DataFrame(data_rows)
        os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
        df.to_csv(CSV_FILE, index=False)
        print(f"[CSV] Exported CSV to {os.path.abspath(CSV_FILE)}")
    else:
        print("[CSV] No data to export.")


# ------------------- Example Usage -------------------
if __name__ == "__main__":
    # Example usage:
    # save_to_hdf5(window_id, window_data, annotator, labels)

    # Export all data to CSV and images
    hdf5_to_csv_export()
