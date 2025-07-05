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
)  # Use a non-interactive backend for Matplotlib in a server environment
import matplotlib.pyplot as plt
import threading
from collections import defaultdict
import io
from threading import Lock

app = Flask(__name__)
# Moved app_start_time to global scope for production server compatibility
app_start_time = time.time()
CORS(app)
app.static_folder = "static"  # Explicitly serve static files

# === Secret Key for Cleanup ===
CLEANUP_API_KEY = "u75rxkreJ7oHpr"

# === YOLO Models ===
MODEL_PATHS = {
    "ripeness": "yolov11nripeness_150epoch.pt",  # Model for ripeness detection
    "tomato": "yolov11ntomato_200epoch.pt"  # Model for tomato detection
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

# === Registry and Logs (Thread-Safe) ===
# Add a lock for thread-safe access to shared data
registry_lock = Lock()
device_registry = {}
zone_devices = defaultdict(list)
zone_disease_count = defaultdict(int)

# === Persistent Device Storage Functions ===
def load_registered_devices():
    """Load registered devices from JSON file"""
    try:
        if os.path.exists(REGISTERED_DEVICES_FILE):
            with open(REGISTERED_DEVICES_FILE, 'r') as f:
                data = json.load(f)
                # Convert back to proper format
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
    """Save registered devices to JSON file"""
    try:
        data = {
            'device_registry': device_registry,
            'zone_devices': dict(zone_devices)  # Convert defaultdict to regular dict
        }
        with open(REGISTERED_DEVICES_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"💾 Saved {len(device_registry)} registered devices to file")
    except Exception as e:
        print(f"❌ Error saving registered devices: {e}")

# Load existing registered devices on startup
load_registered_devices()

# =================================================================
# Background Cleanup Task
# =================================================================
def cleanup_devices():
    """Periodically remove inactive devices (every 5 minutes)"""
    while True:
        time.sleep(300)  # 5 minutes
        now = time.time()
        expired_devices_info = []

        # Use the lock to safely identify expired devices
        with registry_lock:
            for device_id, device in device_registry.items():
                if now - device["last_seen"] > 600:  # 10 minute timeout
                    expired_devices_info.append(
                        {"id": device_id, "zone": device["zone"]}
                    )

        # Perform deletions outside the initial lock to keep the lock duration short
        for device_info in expired_devices_info:
            device_id = device_info["id"]
            zone = device_info["zone"]
            # Use the lock again for the modification part
            with registry_lock:
                if (
                    device_id in device_registry
                ):  # Check if it wasn't re-registered in the meantime
                    del device_registry[device_id]
                    if device_id in zone_devices[zone]:
                        zone_devices[zone].remove(device_id)
                    print(f"🧹 Removed expired device: {device_id} from zone: {zone}")
                    # Save changes to file
                    save_registered_devices()

cleanup_thread = threading.Thread(target=cleanup_devices, daemon=True)
cleanup_thread.start()

# =================================================================
# Helper Functions
# =================================================================
def notify_neighbor(ip):
    """Send notification to neighbor device in a non-blocking way"""
    try:
        requests.post(f"http://{ip}:9000", data="TAKE_PHOTO", timeout=1)
    except Exception as e:
        print(f"⚠️ Neighbor notification failed to {ip}: {str(e)}")

def process_image_with_model(image_path, model_name):
    """Process image and return both detailed detections and simple count format"""
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

        # Detailed detections for internal use
        detections = []
        for box in r.boxes:
            cls = int(box.cls.item())
            name = models[model_name].names[cls]
            conf = box.conf.item()
            bbox = box.xyxy[0].tolist()
            detections.append({"class": name, "confidence": conf, "bbox": bbox})

        # Simple count format for results output
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

# =================================================================
# API Endpoints
# =================================================================
@app.route("/register", methods=["POST"])
def register_device():
    data = request.get_json()
    if not data or not all(k in data for k in ["device_id", "ip", "zone"]):
        return jsonify({"error": "Missing device_id, ip, or zone"}), 400

    device_id, ip, zone = data["device_id"], data["ip"], data["zone"]

    # Use lock for all modifications to shared registries
    with registry_lock:
        if device_id in device_registry:
            old_zone = device_registry[device_id]["zone"]
            if old_zone != zone and device_id in zone_devices[old_zone]:
                zone_devices[old_zone].remove(device_id)

        device_registry[device_id] = {"ip": ip, "zone": zone, "last_seen": time.time()}
        if device_id not in zone_devices[zone]:  # Avoid duplicates
            zone_devices[zone].append(device_id)

    # Save to file after registration
    save_registered_devices()

    print(f"📱 Registered device: {device_id} in zone: {zone} (IP: {ip})")
    return jsonify({"status": "registered"}), 200

@app.route("/predict", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    if not request.form.get("device_id"):
        return jsonify({"error": "Missing device ID"}), 400

    model_name = request.form.get("model_name", "ripeness")
    device_id = request.form.get("device_id")
    # Handle sensor_value sent from ESP32
    sensor_value = request.form.get("sensor_value")
    # Better IP handling for production environments behind a proxy
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if model_name not in models:
        return jsonify({"error": "Invalid model name"}), 400

    command = "OK"
    detections = []
    results = {}  # Simple count format
    image_filename = None
    neighbors_to_notify = []

    try:
        # Use lock for all reads/writes to shared registries
        with registry_lock:
            if device_id not in device_registry:
                command = "REGISTER_AGAIN"
            else:
                # Update device status
                device_registry[device_id]["ip"] = client_ip
                device_registry[device_id]["last_seen"] = time.time()
                zone = device_registry[device_id]["zone"]

                # Process image
                image_file = request.files["image"]
                input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
                image_file.save(input_path)

                image_filename, detections, results = process_image_with_model(
                    input_path, model_name
                )
                if image_filename is None:
                    return jsonify({"error": str(results)}), 500

                # Store latest prediction for this device
                device_predictions[device_id] = {
                    "timestamp": time.time(),
                    "image_filename": image_filename,
                    "results": results,
                    "detections": detections
                }

                # Analyze detections and update zone counts
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

                    # Get list of neighbors to notify
                    for neighbor_id in zone_devices.get(zone, []):
                        if neighbor_id != device_id and neighbor_id in device_registry:
                            neighbors_to_notify.append(
                                device_registry[neighbor_id]["ip"]
                            )

        # Notify neighbors outside the lock to avoid blocking other requests
        for neighbor_ip in neighbors_to_notify:
            threading.Thread(target=notify_neighbor, args=(neighbor_ip,)).start()

        # Log sensor value if present
        if sensor_value:
            print(f"📊 Received sensor value from {device_id}: {sensor_value}")

        # Return in the simple format you requested
        return jsonify(
            {
                "status": "success",
                "device_id": device_id,
                "results": results,  # Simple format: {"Strawberry_ripe": 4, "Strawberry_unripe": 3}
                "image_url": f"/static/{image_filename}" if image_filename else None,
                "command": command,
            }
        ), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/devices", methods=["GET"])
def get_registered_devices():
    """Get all registered devices from file"""
    with registry_lock:
        return jsonify({
            'status': 'success',
            'devices': device_registry,
            'zone_devices': dict(zone_devices)
        }), 200

@app.route("/generate_heatmap", methods=["GET"])
def generate_heatmap():
    """Generate heatmap in-memory and return directly."""
    try:
        with registry_lock:
            # Make a copy to work with, releasing the lock quickly
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

        # Save to a memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)  # Important to free memory

        return send_file(buf, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/cleanup", methods=["POST"])
def cleanup_server_state():
    """
    Resets all in-memory data and deletes generated image files.
    Requires a valid API key in the 'X-API-KEY' header.
    """
    auth_key = request.headers.get("X-API-KEY")
    if auth_key != CLEANUP_API_KEY:
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 403

    try:
        # 1. Clear in-memory data (thread-safe)
        with registry_lock:
            cleared_devices_count = len(device_registry)
            device_registry.clear()
            zone_devices.clear()
            zone_disease_count.clear()

        # 2. Delete the persistent file
        if os.path.exists(REGISTERED_DEVICES_FILE):
            os.remove(REGISTERED_DEVICES_FILE)

        # 3. Delete generated files from 'uploads' and 'static'
        deleted_files_count = 0
        deletion_errors = []
        for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
            for filename in os.listdir(folder):
                # Only delete image files, leave other files (like models) alone
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    try:
                        file_path = os.path.join(folder, filename)
                        os.remove(file_path)
                        deleted_files_count += 1
                    except Exception as e:
                        error_msg = f"Could not delete {file_path}: {str(e)}"
                        deletion_errors.append(error_msg)
                        print(f"⚠️ {error_msg}")

        # 4. Construct and send the response
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

# Store latest predictions for each device
device_predictions = {}

@app.route("/latest/<device_id>", methods=["GET"])
def get_latest_prediction(device_id):
    """
    Get the latest prediction for a specific device.
    Returns photo and disease/ripeness counts in React-friendly format.
    """
    try:
        with registry_lock:
            if device_id not in device_registry:
                return jsonify({"error": "Device not found or not registered"}), 404
            
            if device_id not in device_predictions:
                return jsonify({"error": "No predictions available for this device"}), 404
            
            prediction = device_predictions[device_id]
            device_info = device_registry[device_id]
            
            # Format response for React frontend
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
    """
    Get latest predictions for all devices.
    Perfect for React dashboard showing all device statuses.
    """
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
                
                # Add prediction data if available
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
    # Use lock to get a consistent snapshot of the stats
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