import os
import csv
import cv2
import numpy as np
import json

def save_to_csv_and_video(window_id, window_data, labels_dict, driver_id="driver_1"):
    """
    Save metrics, raw data, annotator labels, and video for each window.
    Video filename = <window_id>.mp4
    """
    base_folder = "drowsiness_data"
    videos_folder = os.path.join(base_folder, "videos", driver_id)
    os.makedirs(videos_folder, exist_ok=True)

    csv_file = os.path.join(base_folder, f"{driver_id}_session.csv")

    # --- Save video ---
    video_filename = f"{window_id}.mp4"
    video_path = os.path.join(videos_folder, video_filename)
    images = window_data.get("images", [])

    if images:
        try:
            first_frame = cv2.imdecode(np.frombuffer(images[0], np.uint8), cv2.IMREAD_COLOR)
            height, width, _ = first_frame.shape
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for img_bytes in images:
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                out.write(frame)
            out.release()
        except Exception as e:
            print(f"[VIDEO] Failed to save video for window {window_id}: {e}")
            video_path = None
    else:
        video_path = None

    # --- Prepare CSV row ---
    row = {
        "window_id": window_id,
        "video": video_filename if video_path else "",
    }

    # Metrics
    metrics = window_data.get("metrics", {})
    for k, v in metrics.items():
        row[f"metric_{k}"] = v

    # Raw data
    raw_data = window_data.get("raw_data", {})
    for k, v in raw_data.items():
        if v is not None:
            row[f"raw_{k}"] = ", ".join(map(str, v))

    # Annotator labels with prefixes
    for annotator, labels in labels_dict.items():
        prefix = annotator.replace(" ", "_")
        for k, v in labels.items():
            if isinstance(v, (list, dict)):
                v = json.dumps(v)
            row[f"{prefix}_{k}"] = v

    # Write to CSV
    write_header = not os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[CSV+VIDEO] Saved window {window_id} data with video='{video_filename}' and annotators={list(labels_dict.keys())}")
