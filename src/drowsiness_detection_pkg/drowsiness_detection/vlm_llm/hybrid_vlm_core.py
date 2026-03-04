#!/usr/bin/env python3
"""
This node develops the system prompt for the VLM. And core instructions.
"""
import os
import cv2
import numpy as np
import threading
import base64
from collections import deque
from datetime import datetime
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor

SYSTEM_PROMPT = """SYSTEM: You are Qwen2.5-VL for automated driver behavior evaluation. 
You will receive 8 consecutive frames from a driver's face, FOLLOWED BY 3 graphs showing EAR, MAR, and BPM over the last 2 seconds.
Produce JSON-only output following exactly this schema:
{
    "clip_id": "string",
    "drowsy": {"label": "yes/no", "confidence": 0.0},
    "behaviors": {
        "yawn": {"detected": false, "confidence": 0.0},
        "eyes_closed": {"detected": false, "confidence": 0.0}
    },
    "occlusion": {"face_occluded": true/false, "reason": "mask/hand/sunglare/other/null"},
    "notes": "Short explanation of why EAR dropped (e.g. driver rubbed eye) or if it's genuine drowsiness."
}"""

USER_PROMPT_TEMPLATE = """You will analyze the following 11 images. The first 8 are consecutive camera frames. The last 3 are data graphs.
Analyze them together to explain why the metrics look anomalous or if the driver is genuinely drowsy.
Clip id: {clip_id}
Answer only in valid JSON."""

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-vl:3b"

class VLMRequestHandler:
    def __init__(self, max_workers=2, output_dir="/root/ws/drowsiness_data/vlm_events"):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VLM")
        self.output_dir = output_dir

    def submit_request(self, pre_encoded_images, clip_id):
        # Fire and forget
        self.executor.submit(self.vlm_request_blocking, pre_encoded_images, clip_id)

    def vlm_request_blocking(self, imgs_b64, clip_id):
        payload = {
            "model": MODEL_NAME,
            "system": SYSTEM_PROMPT,
            "prompt": USER_PROMPT_TEMPLATE.format(clip_id=clip_id),
            "images": imgs_b64,
            "stream": False,
            "format": "json"
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            result = json.loads(resp.json()['response'])
            
            # Save to disk
            out_dir = os.path.join(self.output_dir, "events", clip_id)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "analysis.json"), "w") as f:
                json.dump(result, f, indent=2)
                
            return result
        except Exception as e:
            print(f"VLM Request Failed: {e}")
            return None
