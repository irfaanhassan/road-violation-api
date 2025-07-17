from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load models globally ----------

helmet_model = YOLO('D:/api_folder/road-violation-api/models/helmet.pt')
triple_model = YOLO('D:/api_folder/road-violation-api/models/yolov8n.pt')
wrong_route_model = YOLO("D:/api_folder/road-violation-api/models/wrong_route.pt")
pothole_model = YOLO("D:/api_folder/road-violation-api/models/potholes.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

HELMET_CLASSES = {0: "Plate", 1: "WithHelmet", 2: "WithoutHelmet"}
WRONG_ROUTE_CLASSES = ['Right Side', 'Wrong Side']
#POTHOLE_CLASSES = ['pothole', 'other']

# ---------- Detection functions ----------

def detect_helmet_violation(image_path):
    results = helmet_model(image_path)
    helmet_violations = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 2:
                helmet_violations.append(box)

    detected = len(helmet_violations) > 0
    plates = []

    if detected:
        ocr_results = ocr.ocr(image_path, cls=True)
        if ocr_results and ocr_results[0]:
            for result in ocr_results[0]:
                if len(result) >= 2 and isinstance(result[1], (list, tuple)):
                    plate_text = result[1][0].strip()
                    if plate_text:
                        plates.append(plate_text)

    return {"detected": detected, "plates": plates}


def detect_triple_riding(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = triple_model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()

    persons, motorcycles = [], []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:
            persons.append((x1, y1, x2, y2))
        elif int(cls) == 3:
            motorcycles.append((x1, y1, x2, y2))

    def is_inside(person, moto):
        px, py = (person[0]+person[2])/2, (person[1]+person[3])/2
        return moto[0] <= px <= moto[2] and moto[1] <= py <= moto[3]

    for moto in motorcycles:
        count = sum(1 for person in persons if is_inside(person, moto))
        if count > 2:
            return {"detected": True}

    return {"detected": False}


def detect_wrong_route(image_path):
    image = cv2.imread(image_path)
    results = wrong_route_model(image)
    for result in results:
        for box in result.boxes:
            if WRONG_ROUTE_CLASSES[int(box.cls[0])].lower() == "wrong side":
                return {"detected": True}
    return {"detected": False}


#def detect_pothole(image_path):
   # image = cv2.imread(image_path)
    #results = pothole_model(image)
    #for result in results:
     #   for box in result.boxes:
      #      if POTHOLE_CLASSES[int(box.cls[0])] == "pothole":
       #         return {"detected": True}
    #return {"detected": False}

# ---------- Unified predict endpoint ----------

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            image_bytes = await file.read()
            temp_image.write(image_bytes)
            temp_path = temp_image.name

        result = {
            "helmet_violation": detect_helmet_violation(temp_path),
            "triple_riding": detect_triple_riding(temp_path),
            "wrong_route": detect_wrong_route(temp_path),
            #"pothole": detect_pothole(temp_path)
        }

        os.remove(temp_path)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
