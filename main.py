from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = FastAPI()

# ---------- Load all models once globally ----------

# Helmet detection model & class names
helmet_model = YOLO('C:/Users/admin/OneDrive/Desktop/New folder/app/models/helmet.pt')
HELMET_CLASSES = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet"}

# Triple riding model
triple_model = YOLO('C:/Users/admin/OneDrive/Desktop/New folder/app/models/yolov8n.pt')

# Wrong route model
wrong_route_model = YOLO("C:/Users/admin/OneDrive/Desktop/New folder/app/models/wrong_route.pt")
WRONG_ROUTE_CLASSES = ['Right Side', 'Wrong Side']

# Pothole model
pothole_model = YOLO("C:/Users/admin/OneDrive/Desktop/New folder/app/models/potholes.pt")
POTHOLE_CLASSES = ['pothole', 'other']

# PaddleOCR for number plate reading (disable GPU if not available)
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# ----------------- Detection Functions ------------------

def detect_helmet_violation(image_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(image_bytes)
        temp_path = temp_image.name

    results = helmet_model(temp_path)
    helmet_violations = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 2:  # WithoutHelmet
                helmet_violations.append(box)

    helmet_violation_detected = len(helmet_violations) > 0
    response = {"helmet_violation": helmet_violation_detected}

    if helmet_violation_detected:
        ocr_results = ocr.ocr(temp_path, cls=True)
        print("OCR Results:", ocr_results)
        plates = []

        if ocr_results and ocr_results[0] is not None:
            for result in ocr_results[0]:
                if len(result) >= 2 and isinstance(result[1], (list, tuple)) and len(result[1]) >= 1:
                    plate_text = result[1][0].strip()
                    if plate_text:
                        plates.append(plate_text)

        if plates:
            response["number_plates"] = plates

    os.remove(temp_path)
    return response

def detect_triple_riding(image_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(image_bytes)
        temp_path = temp_image.name

    image = cv2.imread(temp_path)
    if image is None:
        os.remove(temp_path)
        raise ValueError("Could not read the uploaded image.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = triple_model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    persons = []
    motorcycles = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:
            persons.append((x1, y1, x2, y2))
        elif int(cls) == 3:
            motorcycles.append((x1, y1, x2, y2))

    def is_center_inside(person, moto):
        px_center = (person[0] + person[2]) / 2
        py_center = (person[1] + person[3]) / 2
        return (moto[0] <= px_center <= moto[2]) and (moto[1] <= py_center <= moto[3])

    triple_riding_detected = False
    for moto in motorcycles:
        count = 0
        for person in persons:
            if is_center_inside(person, moto):
                count += 1
        if count > 2:
            triple_riding_detected = True
            break

    os.remove(temp_path)

    return {
        "triple_riding_detected": triple_riding_detected
    }


def detect_wrong_route(image_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(image_bytes)
        temp_path = temp_image.name

    image = cv2.imread(temp_path)
    results = wrong_route_model(image)

    wrong_route_detected = False
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if WRONG_ROUTE_CLASSES[cls_id].lower() == "wrong side":
                wrong_route_detected = True
                break

    os.remove(temp_path)
    return {
        "wrong_route_detected": wrong_route_detected
    }


def detect_pothole(image_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        temp_image.write(image_bytes)
        temp_path = temp_image.name

    image = cv2.imread(temp_path)
    results = pothole_model(image)

    pothole_detected = False
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = POTHOLE_CLASSES[cls_id]
            if label == "pothole":
                pothole_detected = True
                break

    os.remove(temp_path)

    return {
        "pothole_detected": pothole_detected
    }

# ----------------- API Endpoints ------------------

@app.post("/detect_helmet")
async def api_detect_helmet(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = detect_helmet_violation(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_triple_riding")
async def api_detect_triple_riding(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = detect_triple_riding(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_wrong_route")
async def api_detect_wrong_route(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = detect_wrong_route(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_pothole")
async def api_detect_pothole(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = detect_pothole(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
