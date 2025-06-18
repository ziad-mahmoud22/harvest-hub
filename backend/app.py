
from flask import Flask, request, jsonify
import requests
import base64
import io
from PIL import Image
import os
import json
import uuid
import traceback
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  

MODEL_PATHS = {
    "ripeness": "yolov11nripeness_150epoch.pt",  # Model for ripeness detection
    "tomato": "yolov11ntomato_200epoch.pt"  # Model for tomato detection
}

models = {}
for model_name, model_path in MODEL_PATHS.items():
    try:
        models[model_name] = YOLO(model_path)
        print(f"Model '{model_name}' loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        models[model_name] = None

# Ensure the upload and output folders exist
UPLOAD_FOLDER = r'C:\Users\ziadm\Desktop\projects\gp\uploads'
OUTPUT_FOLDER = r'C:\Users\ziadm\Desktop\projects\gp\outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_image_with_model(image_path, model_name):
 
    if model_name not in models or models[model_name] is None:
        return None, f"Model '{model_name}' not loaded"
    
    try:
        # perdection
        results = models[model_name].predict(image_path)
        r = results[0]

        # gen image
        im_array = r.plot()  # BGR numpy array
        im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB
        
        # save image to bytes
        output_image_bytes = io.BytesIO()
        im.save(output_image_bytes, format="JPEG")
        output_image_bytes.seek(0)
        image_data = output_image_bytes.read()

        # Generate text output (class counts)
        classes = r.boxes.cls.tolist()
        class_counts = {}
        names = models[model_name].names
        for c in classes:
            class_name = names[int(c)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        text_output = json.dumps(class_counts, indent=4) 

        return image_data, text_output
    except Exception as e:
        print(f"Error during prediction with model '{model_name}': {e}")
        traceback.print_exc()
        return None, str(e)

@app.route('/predict', methods=['POST'])
def process_image():
    """
    Handle image upload, process it, save the outputs, and return results.
    """
    print("Received request to /predict")
    try:
        if 'image' not in request.files:
            print("No image file found in request")
            return jsonify({'error': 'No image provided'}), 400
        
        model_name = request.form.get('model_name', 'ripeness')
        if model_name not in models:
            print(f"Invalid model name: {model_name}")
            return jsonify({'error': f"Invalid model name: {model_name}"}), 400
        
        image_file = request.files['image']
        
        # Create a unique base filename to link input and outputs
        base_filename = str(uuid.uuid4())
        input_filename = f"{base_filename}.jpg"
        input_image_path = os.path.join(UPLOAD_FOLDER, input_filename)
        image_file.save(input_image_path)
        print(f"Image saved to: {input_image_path}")

        
        processed_image_bytes, text_output = process_image_with_model(input_image_path, model_name)
        if processed_image_bytes is None:
            print(f"Image processing failed: {text_output}")
            return jsonify({'error': text_output}), 500

       
        output_image_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_processed.jpg")
        output_text_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_results.json")

        #save processed image
        with open(output_image_path, "wb") as f:
            f.write(processed_image_bytes)
        print(f"Processed image saved to: {output_image_path}")

        # save text output as jason
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(text_output)
        print(f"Text results saved to: {output_text_path}")

        processed_image_b64 = base64.b64encode(processed_image_bytes).decode('utf-8')

    #     # Send the results to the Express server
    #     response = requests.post(
    #         'http://localhost:3000/results',
    #         json={'image': processed_image_b64, 'text': text_output, 'image_path': input_image_path},
    #         timeout=10
    #     )
    #     response.raise_for_status()

    #     return jsonify(response.json()), response.status_code
    # except Exception as e:
    #     print(f"Unexpected error: {e}")
    #     traceback.print_exc()
    #     return jsonify({'error': f'An unexpected error occurred: {e}'}), 500
        return jsonify({
                'status': 'success',
                'message': 'Image processed successfully.',
                'processed_image': processed_image_b64,
                'results': json.loads(text_output) # Convert text back to a real JSON object
            }), 200
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

@app.route('/', methods=['GET'])
def health_check():
    """
    Health check
    """
    return jsonify({"status": "Server is running"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)
