from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import io
import json
import uuid
import base64
import traceback
from PIL import Image
import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app)

# === YOLO Models ===
MODEL_PATHS = {
    "ripeness": "yolov11nripeness_150epoch.pt",
    "tomato": "yolov11ntomato_200epoch.pt"
}

models = {}
for model_name, model_path in MODEL_PATHS.items():
    try:
        models[model_name] = YOLO(model_path)
        print(f"✅ Loaded model: {model_name}")
    except Exception as e:
        print(f"❌ Error loading model '{model_name}': {e}")
        models[model_name] = None

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Registry and Logs ===
device_registry = {}

# Virtual Nodes with Locations (MAC and Grid Position)
VIRTUAL_NODES = {
    "MAC1": {"neighbors": ["MAC2", "MAC4"], "position": (0, 0)},
    "MAC2": {"neighbors": ["MAC1", "MAC3"], "position": (0, 1)},
    "MAC3": {"neighbors": ["MAC2", "MAC6"], "position": (0, 2)},
    "MAC4": {"neighbors": ["MAC1", "MAC5"], "position": (1, 0)},
    "MAC5": {"neighbors": ["MAC4", "MAC6"], "position": (1, 1)},
    "MAC6": {"neighbors": ["MAC3", "MAC5"], "position": (1, 2)}
}

# Simulated device positions for heatmap
POSITION_MAP = {mac: data["position"] for mac, data in VIRTUAL_NODES.items()}

# Log of results
heatmap_data = []

@app.route('/register', methods=['POST'])
def register_device():
    data = request.get_json()
    mac = data.get("device_id")
    ip = data.get("ip")

    if mac and ip:
        device_registry[mac] = {"ip": ip, "last_seen": time.time()}
        return jsonify({"status": "registered"}), 200
    else:
        return jsonify({"error": "Missing MAC or IP"}), 400

def process_image_with_model(image_path, model_name):
    if model_name not in models or models[model_name] is None:
        return None, f"Model '{model_name}' not loaded"

    try:
        results = models[model_name].predict(image_path)
        r = results[0]
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

        output_image_bytes = io.BytesIO()
        im.save(output_image_bytes, format="JPEG")
        output_image_bytes.seek(0)
        image_data = output_image_bytes.read()

        class_counts = {}
        for cls in r.boxes.cls.tolist():
            name = models[model_name].names[int(cls)]
            class_counts[name] = class_counts.get(name, 0) + 1

        return image_data, json.dumps(class_counts, indent=4)
    except Exception as e:
        traceback.print_exc()
        return None, str(e)

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400

        model_name = request.form.get("model_name", "ripeness")
        mac_address = request.form.get("device_id")

        if model_name not in models:
            return jsonify({'error': 'Invalid model name'}), 400

        image_file = request.files['image']
        base_filename = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, base_filename + ".jpg")
        image_file.save(input_path)

        image_bytes, json_output = process_image_with_model(input_path, model_name)
        if image_bytes is None:
            return jsonify({'error': json_output}), 500

        detected_classes = list(json.loads(json_output).keys())
        disease_score = sum([1 for cls in detected_classes if "disease" in cls.lower()])

        if mac_address in POSITION_MAP:
            x, y = POSITION_MAP[mac_address]
            heatmap_data.append((x, y, disease_score))

        # Trigger neighbors if disease detected
        if disease_score > 0:
            for neighbor in VIRTUAL_NODES.get(mac_address, {}).get("neighbors", []):
                if neighbor in device_registry:
                    neighbor_ip = device_registry[neighbor]["ip"]
                    try:
                        requests.post(f"http://{neighbor_ip}:9000", data="TAKE_PHOTO\n", timeout=3)
                    except:
                        pass

        return jsonify({
            "status": "success",
            "device_id": mac_address,
            "results": json.loads(json_output)
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_heatmap', methods=['GET'])
def generate_heatmap():
    try:
        grid = np.zeros((2, 3))  # Based on 2x3 grid from POSITION_MAP
        for x, y, score in heatmap_data:
            grid[x, y] += score

        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.colorbar()
        heatmap_file = os.path.join(OUTPUT_FOLDER, "heatmap.png")
        plt.title("Disease Heatmap")
        plt.savefig(heatmap_file)
        plt.close()

        return jsonify({"status": "heatmap generated", "path": heatmap_file}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "Server is running"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
