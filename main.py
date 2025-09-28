import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from ultralytics import YOLO
import tempfile
import os
import asyncio
import mysql.connector
from datetime import datetime, timedelta
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NoxyAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {
    "Noxy": "noxy.pt",
}

db_config = {
    'host': '188.127.241.8',
    'user': 'gs107231',
    'password': 'CnS2p1qh9h',
    'database': 'gs107231'
}

loaded_models = {}
for name, model_path in MODELS.items():
    try:
        loaded_models[name] = YOLO(model_path)
        print(f"Model {name} loaded successfully")
    except Exception as e:
        print(f"Error loading {name}: {e}")

def get_db_connection():
    return mysql.connector.connect(**db_config)

def create_tables():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                login VARCHAR(255) UNIQUE NOT NULL,
                api_key VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                email VARCHAR(255),
                register_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                subscription_end DATETIME,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Tables created successfully")
    except Exception as e:
        print(f"Error creating tables: {e}")

create_tables()

async def verify_api_key(api_key: str = Header(...), login: str = Header(...)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT * FROM users WHERE login = %s AND api_key = %s AND is_active = TRUE",
            (login, api_key)
        )
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key or login")
        
        current_time = datetime.now()
        if user['subscription_end'] and user['subscription_end'] < current_time:
            raise HTTPException(
                status_code=403, 
                detail="Subscription expired. Please renew: https://noxy.com"
            )
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def extract_captcha_from_yolo(results, confidence_threshold=0.3):
    if not results or len(results) == 0:
        return "", 0.0

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return "", 0.0

    boxes = result.boxes.xywh.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    valid_indices = np.where(confidences >= confidence_threshold)[0]

    if len(valid_indices) == 0:
        return "", 0.0

    characters = []
    total_confidence = 0.0

    for idx in valid_indices:
        x_center = boxes[idx][0]
        class_id = int(classes[idx])
        character = str(class_id)
        characters.append((x_center, character, confidences[idx]))
        total_confidence += confidences[idx]

    characters.sort(key=lambda x: x[0])
    captcha_text = "".join([char[1] for char in characters])
    avg_confidence = total_confidence / len(characters) if characters else 0.0

    return captcha_text, avg_confidence

async def predict_with_model(model, image_path):
    try:
        results = model.predict(
            image_path,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            verbose=False
        )

        captcha_text, confidence = extract_captcha_from_yolo(results)

        return {
            "success": bool(captcha_text),
            "captcha_text": captcha_text,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "success": False,
            "captcha_text": "",
            "confidence": 0.0,
            "error": str(e)
        }

@app.post("/captcha")
async def solve_captcha(
    image: UploadFile = File(...),
    user: dict = Depends(verify_api_key)
):
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await image.read()
            temp_file.write(content)
            image_path = temp_file.name

        try:
            best_result = {"captcha_text": "", "confidence": 0.0}

            for model_name, model in loaded_models.items():
                result = await predict_with_model(model, image_path)
                print(f"Model {model_name}: {result['captcha_text']}")

                if result["success"] and result["confidence"] > best_result["confidence"]:
                    best_result = result

            if not best_result["captcha_text"]:
                return {"captcha": "0000"}

            return {"captcha": best_result["captcha_text"]}

        finally:
            os.unlink(image_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/status")
async def server_status(user: dict = Depends(verify_api_key)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT COUNT(*) as total_users FROM users")
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute("SELECT COUNT(*) as active_users FROM users WHERE is_active = TRUE")
        active_users = cursor.fetchone()['active_users']
        
        cursor.close()
        conn.close()
        
        models_status = {}
        for name, model in loaded_models.items():
            models_status[name] = "loaded" if model else "error"
        
        return {
            "server_status": "online",
            "total_users": total_users,
            "active_users": active_users,
            "models_status": models_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.get("/")
async def root():
    return {"message": "NoxyAPI is working!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

