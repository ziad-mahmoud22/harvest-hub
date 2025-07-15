import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import traceback
from PIL import Image
import requests
import time
import matplotlib
import json
if 'torch' in globals():
    import importlib; importlib.reload(globals()['torch'])
    
matplotlib.use(
    "Agg"
) 
import matplotlib.pyplot as plt
import threading
from collections import defaultdict
import io
from threading import Lock

app = Flask(__name__)
app_start_time = time.time()
CORS(app)
app.static_folder = "static"  

CLEANUP_API_KEY = "u75rxkreJ7oHpr"

MODEL_PATHS = {
    "ripeness": "yolov11nripeness_150epoch.pt",
    "tomato": "yolov11ntomato_200epoch.pt"
}

disease_list = [
    "early blight",
    "late blight",
    "leaf miner",
    "leaf mold",
    "mosaic virus",
    "septoria",
    "spider mites",
    "yellow leaf curl virus",
]

models = {}
for model_name, model_path in MODEL_PATHS.items():
    try:
        models[model_name] = YOLO(model_path)
        print(f"✅ Loaded model: {model_name}")
    except Exception as e:
        print(f"❌ Error loading model '{model_name}': {e}")
        models[model_name] = None

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
REGISTERED_DEVICES_FILE = "registered_devices.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

registry_lock = Lock()
device_registry = {}
zone_devices = defaultdict(list)
zone_disease_count = defaultdict(int)

def load_registered_devices():
    try:
        if os.path.exists(REGISTERED_DEVICES_FILE):
            with open(REGISTERED_DEVICES_FILE, 'r') as f:
                data = json.load(f)
                
                global device_registry, zone_devices
                device_registry = data.get('device_registry', {})
                zone_devices_dict = data.get('zone_devices', {})
                zone_devices = defaultdict(list)
                for zone, devices in zone_devices_dict.items():
                    zone_devices[zone] = devices
                print(f"📂 Loaded {len(device_registry)} registered devices from file")
                return True
    except Exception as e:
        print(f"❌ Error loading registered devices: {e}")
    return False

def save_registered_devices():
    try:
        data = {
            'device_registry': device_registry,
            'zone_devices': dict(zone_devices)         }
        with open(REGISTERED_DEVICES_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"💾 Saved {len(device_registry)} registered devices to file")
    except Exception as e:
        print(f"❌ Error saving registered devices: {e}")

load_registered_devices()


def cleanup_devices():
    while True:
        time.sleep(300)
        now = time.time()
        expired_devices_info = []

        with registry_lock:
            for device_id, device in device_registry.items():
                if now - device["last_seen"] > 600:
                    expired_devices_info.append(
                        {"id": device_id, "zone": device["zone"]}
                    )

        for device_info in expired_devices_info:
            device_id = device_info["id"]
            zone = device_info["zone"]
            with registry_lock:
                if (
                    device_id in device_registry
                ):
                    del device_registry[device_id]
                    if device_id in zone_devices[zone]:
                        zone_devices[zone].remove(device_id)
                    print(f"🧹 Removed expired device: {device_id} from zone: {zone}")
                    
                    save_registered_devices()

cleanup_thread = threading.Thread(target=cleanup_devices, daemon=True)
cleanup_thread.start()

@app.route("/register", methods=["POST"])
def register_device():
    data = request.get_json()
    if not data or not all(k in data for k in ["device_id", "ip", "zone"]):
        return jsonify({"error": "Missing device_id, ip, or zone"}), 400

    device_id, ip, zone = data["device_id"], data["ip"], data["zone"]

    with registry_lock:
        if device_id in device_registry:
            old_zone = device_registry[device_id]["zone"]
            if old_zone != zone and device_id in zone_devices[old_zone]:
                zone_devices[old_zone].remove(device_id)

        device_registry[device_id] = {"ip": ip, "zone": zone, "last_seen": time.time()}
        if device_id not in zone_devices[zone]:
            zone_devices[zone].append(device_id)

    save_registered_devices()

    print(f"📱 Registered device: {device_id} in zone: {zone} (IP: {ip})")
    return jsonify({"status": "registered"}), 200

def notify_neighbor(ip):
    try:
        requests.post(f"http://{ip}:9000", data="TAKE_PHOTO", timeout=1)
    except Exception as e:
        print(f"⚠️ Neighbor notification failed to {ip}: {str(e)}")

def process_image_with_model(image_path, model_name):
    if model_name not in models or not models[model_name]:
        return None, None, f"Model '{model_name}' not loaded"
    try:
        results = models[model_name].predict(image_path)
        r = results[0]
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join(STATIC_FOLDER, output_filename)
        im.save(output_path)

        detections = []
        for box in r.boxes:
            cls = int(box.cls.item())
            name = models[model_name].names[cls]
            conf = box.conf.item()
            bbox = box.xyxy[0].tolist()
            detections.append({"class": name, "confidence": conf, "bbox": bbox})

        classes = r.boxes.cls.tolist()
        class_counts = {}
        names = models[model_name].names
        for c in classes:
            class_name = names[int(c)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return output_filename, detections, class_counts
    except Exception as e:
        traceback.print_exc()
        return None, None, str(e)

@app.route("/predict", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    if not request.form.get("device_id"):
        return jsonify({"error": "Missing device ID"}), 400

    model_name = request.form.get("model_name", "ripeness")
    device_id = request.form.get("device_id")
    sensor_value = request.form.get("sensor_value")
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if model_name not in models:
        return jsonify({"error": "Invalid model name"}), 400

    command = "OK"
    detections = []
    results = {}
    image_filename = None
    neighbors_to_notify = []

    try:
        with registry_lock:
            if device_id not in device_registry:
                command = "REGISTER_AGAIN"
            else:
                device_registry[device_id]["ip"] = client_ip
                device_registry[device_id]["last_seen"] = time.time()
                zone = device_registry[device_id]["zone"]

                image_file = request.files["image"]
                input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
                image_file.save(input_path)

                image_filename, detections, results = process_image_with_model(
                    input_path, model_name
                )
                if image_filename is None:
                    return jsonify({"error": str(results)}), 500

                device_predictions[device_id] = {
                    "timestamp": time.time(),
                    "image_filename": image_filename,
                    "results": results,
                    "detections": detections
                }

                disease_score = sum(
                    1
                    for det in detections
                    if any(d in det["class"].lower() for d in disease_list)
                )
                if disease_score > 0:
                    zone_disease_count[zone] += 1
                    print(
                        f"🚨 Disease detected in zone {zone} by {device_id} (Score: {disease_score})"
                    )

                    for neighbor_id in zone_devices.get(zone, []):
                        if neighbor_id != device_id and neighbor_id in device_registry:
                            neighbors_to_notify.append(
                                device_registry[neighbor_id]["ip"]
                            )

        for neighbor_ip in neighbors_to_notify:
            threading.Thread(target=notify_neighbor, args=(neighbor_ip,)).start()

        if sensor_value:
            print(f"📊 Received sensor value from {device_id}: {sensor_value}")

        return jsonify(
            {
                "status": "success",
                "device_id": device_id,
                "results": results,
                "image_url": f"/static/{image_filename}" if image_filename else None,
                "command": command,
            }
        ), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/devices", methods=["GET"])
def get_registered_devices():
    with registry_lock:
        return jsonify({
            'status': 'success',
            'devices': device_registry,
            'zone_devices': dict(zone_devices)
        }), 200

@app.route("/generate_heatmap", methods=["GET"])
def generate_heatmap():
    try:
        with registry_lock:
            local_counts = dict(zone_disease_count)

        if not local_counts:
            return jsonify({"error": "No disease data available"}), 404

        zones = list(local_counts.keys())
        counts = [local_counts[z] for z in zones]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(zones, counts, color="red")
        ax.set_xlabel("Zone")
        ax.set_ylabel("Disease Count")
        ax.set_title("Disease Distribution by Zone")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/cleanup", methods=["POST"])
def cleanup_server_state():
    auth_key = request.headers.get("X-API-KEY")
    if auth_key != CLEANUP_API_KEY:
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 403

    try:
        with registry_lock:
            cleared_devices_count = len(device_registry)
            device_registry.clear()
            zone_devices.clear()
            zone_disease_count.clear()

        if os.path.exists(REGISTERED_DEVICES_FILE):
            os.remove(REGISTERED_DEVICES_FILE)

        deleted_files_count = 0
        deletion_errors = []
        for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    try:
                        file_path = os.path.join(folder, filename)
                        os.remove(file_path)
                        deleted_files_count += 1
                    except Exception as e:
                        error_msg = f"Could not delete {file_path}: {str(e)}"
                        deletion_errors.append(error_msg)
                        print(f"⚠️ {error_msg}")

        response_data = {
            "status": "success",
            "message": "Server state and generated files have been cleared.",
            "data_cleared": {
                "devices": cleared_devices_count,
                "zone_device_lists": "cleared",
                "zone_disease_counts": "cleared",
                "persistent_file": "deleted"
            },
            "files_deleted": deleted_files_count,
            "errors": deletion_errors if deletion_errors else "none",
        }
        print("🧹 Server state cleaned up via /cleanup endpoint.")
        return jsonify(response_data), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(
            {"error": f"An unexpected error occurred during cleanup: {str(e)}"}
        ), 500

device_predictions = {}

@app.route("/latest/<device_id>", methods=["GET"])
def get_latest_prediction(device_id):
    try:
        with registry_lock:
            if device_id not in device_registry:
                return jsonify({"error": "Device not found or not registered"}), 404
            
            if device_id not in device_predictions:
                return jsonify({"error": "No predictions available for this device"}), 404
            
            prediction = device_predictions[device_id]
            device_info = device_registry[device_id]
            
            response_data = {
                "device_id": device_id,
                "zone": device_info["zone"],
                "timestamp": prediction["timestamp"],
                "image_url": f"/static/{prediction['image_filename']}" if prediction['image_filename'] else None,
                "results": prediction["results"],
                "summary": {
                    "total_detections": sum(prediction["results"].values()),
                    "categories": list(prediction["results"].keys())
                }
            }
            
            return jsonify(response_data), 200
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/all_latest", methods=["GET"])
def get_all_latest_predictions():
    try:
        with registry_lock:
            all_predictions = []
            
            for device_id, device_info in device_registry.items():
                device_data = {
                    "device_id": device_id,
                    "zone": device_info["zone"],
                    "ip": device_info["ip"],
                    "last_seen": device_info["last_seen"],
                    "status": "online" if time.time() - device_info["last_seen"] < 600 else "offline"
                }
                
                if device_id in device_predictions:
                    prediction = device_predictions[device_id]
                    device_data.update({
                        "has_prediction": True,
                        "prediction_timestamp": prediction["timestamp"],
                        "image_url": f"/static/{prediction['image_filename']}" if prediction['image_filename'] else None,
                        "results": prediction["results"],
                        "summary": {
                            "total_detections": sum(prediction["results"].values()),
                            "categories": list(prediction["results"].keys())
                        }
                    })
                else:
                    device_data.update({
                        "has_prediction": False,
                        "prediction_timestamp": None,
                        "image_url": None,
                        "results": {},
                        "summary": {
                            "total_detections": 0,
                            "categories": []
                        }
                    })
                
                all_predictions.append(device_data)
            
            return jsonify({
                "status": "success",
                "total_devices": len(all_predictions),
                "devices": all_predictions
            }), 200
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    with registry_lock:
        stats = {
            "device_count": len(device_registry),
            "zones": {
                zone: len(devices) for zone, devices in zone_devices.items() if devices
            },
            "disease_counts": dict(zone_disease_count),
        }
    return jsonify(stats)

@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "Server is running",
            "models_loaded": [
                name for name, model in models.items() if model is not None
            ],
            "uptime_seconds": time.time() - app_start_time,
        }
    ), 200

if __name__ == "__main__":
    print("🚀 Starting Flask server with:")
    print(f" - Models: {list(models.keys())}")
    print(f" - Upload folder: {UPLOAD_FOLDER}")
    print(f" - Static folder: {STATIC_FOLDER}")
    print(f" - Registered devices file: {REGISTERED_DEVICES_FILE}")
    app.run(host="0.0.0.0", port=5000)
