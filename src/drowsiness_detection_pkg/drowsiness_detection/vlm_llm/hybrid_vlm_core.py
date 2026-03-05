#!/usr/bin/env python3
"""
This node develops the system prompt for the VLM, handles requests, 
and saves the requested 11 frames (8 camera + 3 graphs) locally.
"""

import os
import base64
from datetime import datetime
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
  "occlusion": {"face_occluded": true/false, "reason": "mask/hand/sunglasses/other/null"},
  "notes": "Short explanation of why EAR dropped (e.g. driver rubbed eye) or if it's genuine drowsiness."
}"""

USER_PROMPT_TEMPLATE = """You will analyze the following 11 images. The first 8 are consecutive camera frames. The last 3 are data graphs.
Analyze them together to explain why the metrics look anomalous or if the driver is genuinely drowsy.
Clip id: {clip_id}
Answer only in valid JSON."""

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL_NAME = "qwen2.5-vl:3b"

class VLMCore:
    def __init__(self, logger, max_workers=2, output_dir="/root/ws/drowsiness_data/vlm_events"):
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VLM")
        self.output_dir = output_dir

    def _save_images_to_disk(self, imgs_b64, clip_id):
        """Decodes the base64 images and saves them into the event directory."""
        event_dir = os.path.join(self.output_dir, "events", clip_id)
        os.makedirs(event_dir, exist_ok=True)
        
        for i, b64_str in enumerate(imgs_b64):
            try:
                img_data = base64.b64decode(b64_str)
                # First 8 are frames, last 3 are graphs
                if i < 8:
                    filename = f"frame_{i}.jpg"
                elif i == 8:
                    filename = "graph_EAR.png"
                elif i == 9:
                    filename = "graph_MAR.png"
                elif i == 10:
                    filename = "graph_BPM.png"
                else:
                    filename = f"extra_{i}.png"
                    
                filepath = os.path.join(event_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(img_data)
            except Exception as e:
                self.logger.error(f"Failed to save image {i} for clip {clip_id}: {e}")

    def query_qwen_vl(self, imgs_b64, prompt_text):
        """Blocking call to Ollama. Used by HybridVLMNode thread."""
        clip_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the 11 images (8 frames + 3 graphs) locally before querying
        self._save_images_to_disk(imgs_b64, clip_id)

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
            result = json.loads(resp.json()["response"])
            
            # Save the JSON analysis next to the images
            out_dir = os.path.join(self.output_dir, "events", clip_id)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "analysis.json"), "w") as f:
                json.dump(result, f, indent=2)
                
            return result
        except Exception as e:
            self.logger.error(f"VLM Request Failed: {e}")
            return None
