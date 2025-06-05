from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import torch
import ultralytics
import os
import json
import gdown

# Model Configuration
MODEL_PATH = 'model_- 4 june 2025 2_10.pt'
CLASS_NAMES = ['person']
CONFIDENCE_THRESHOLD = 0.7
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Google Drive model download (if model is large)
def download_model_if_needed():
    """Download model from Google Drive if not exists locally"""
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        # Replace 'YOUR_GOOGLE_DRIVE_FILE_ID' with actual file ID
        # file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
        # gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH)
        print("Please upload your model file or configure Google Drive download")
        return False
    return True

def load_model():
    """Load and initialize the YOLO model"""
    try:
        if not download_model_if_needed():
            raise FileNotFoundError("Model file not available")
            
        model = ultralytics.YOLO(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print(f"Model '{os.path.basename(MODEL_PATH)}' loaded successfully on {DEVICE}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found")
    except ImportError:
        raise ImportError("Ultralytics library not found. Install with: pip install ultralytics")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def perform_inference(image_pil):
    """Perform object detection inference on the image"""
    results = model(image_pil)
    boxes, labels, scores = [], [], []

    for result in results:
        for *xyxy, conf, cls in result.boxes.data.cpu().numpy():
            if conf >= CONFIDENCE_THRESHOLD:
                boxes.append(xyxy)
                labels.append(int(cls))
                scores.append(float(conf))

    return np.array(boxes), np.array(labels), np.array(scores)

def draw_detections(image_pil, boxes, labels, scores, class_names):
    """Draw bounding boxes and labels on the image"""
    draw = ImageDraw.Draw(image_pil)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        print("Arial font not found, using default font")

    for i in range(len(boxes)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, boxes[i])
            class_name = class_names[labels[i]] if labels[i] < len(class_names) else f"Unknown ({labels[i]})"
            display_text = f"{class_name}: {scores[i]:.2f}"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Calculate text background position
            text_bbox = draw.textbbox((x1, y1), display_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_bg_y1 = max(0, y1 - text_height - 5)

            # Draw text background and text
            draw.rectangle([x1, text_bg_y1, x1 + text_width + 5, y1], fill="red")
            draw.text((x1 + 2, text_bg_y1 + 2), display_text, fill="white", font=font)

    return image_pil

# Initialize FastAPI app
app = FastAPI(
    title="Person Detection API",
    description="Professional API for person detection in images using YOLOv11",
    version="1.0.0"
)

# Initialize model on startup
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        model = load_model()
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        model = None

@app.post("/detect_persons/")
async def detect_persons(file: UploadFile = File(...)):
    """
    Detect persons in uploaded image and return processed image with detection data in headers
    """
    global model
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image")

    try:
        # Read and process image
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Perform detection
        boxes, labels, scores = perform_inference(image_pil)

        # Prepare detection data
        detected_objects = []
        for i in range(len(boxes)):
            if scores[i] >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, boxes[i])
                class_name = CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else f"Unknown ({labels[i]})"
                detected_objects.append({
                    "class": class_name,
                    "confidence": float(scores[i]),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

        # Draw detections on image
        processed_image = image_pil.copy()
        if len(boxes) > 0:
            processed_image = draw_detections(processed_image, boxes, labels, scores, CLASS_NAMES)

        # Convert image to bytes
        img_buffer = io.BytesIO()
        processed_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        # Prepare response data
        detection_data = {
            "message": "Detection completed successfully" if detected_objects else "No persons detected",
            "detections": detected_objects,
            "total_detections": len(detected_objects)
        }

        # Return image with detection data in headers
        return StreamingResponse(
            io.BytesIO(img_buffer.getvalue()),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=person_detection_result.png",
                "X-Detection-Data": json.dumps(detection_data),
                "X-Total-Detections": str(len(detected_objects)),
                "X-Model-Confidence": str(CONFIDENCE_THRESHOLD)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection processing failed: {str(e)}")

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "service": "Person Detection API",
        "version": "1.0.0",
        "model": "YOLOv11",
        "endpoint": "/detect_persons/",
        "description": "Upload an image to detect persons. Returns processed image with detection data in response headers.",
        "status": "Model loaded" if model is not None else "Model not loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
