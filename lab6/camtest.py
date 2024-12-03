from flask import Flask, jsonify
from ultralytics import YOLO
import cv2
import requests
from datetime import datetime
import pytz

app = Flask(__name__)
model = YOLO("yolov8n.pt") 


SIMULATED_IMAGE_PATH = "/home/ubuntu20/lab6/football.jpeg"
DETECTED_IMAGE_PATH = "/home/ubuntu20/lab6/detected_football.jpeg"


API_KEY = "9f18703d01ed26a41982b6e32f5803c4" 
LATITUDE = 40.1164  
LONGITUDE = -88.2434 

@app.route("/detect_status", methods=["GET"])
def detect_status():
    try:
        chicago_tz = pytz.timezone("America/Chicago")
        current_time = datetime.now(chicago_tz)
        current_hour = current_time.hour

        if current_hour < 6 or current_hour >= 20:
            return jsonify({"status": "closed"})

        weather_data = get_weather_data(LATITUDE, LONGITUDE)
        if not weather_data:
            return jsonify({"error": "Failed to fetch weather data."})

        temperature = weather_data.get("temp")
        condition = weather_data.get("condition")
        if (
            temperature < -10 or temperature > 40 or
            "snow" in condition.lower() or
            "rain" in condition.lower() or
            "wind" in condition.lower()
        ):
            return jsonify({"status": "unavailable (weather)"})

        image = cv2.imread(SIMULATED_IMAGE_PATH)
        if image is None:
            return jsonify({"error": "Failed to load image. Check the file path."})

        results = model(image)
        person_count = sum(1 for obj in results[0].boxes.data if obj[5] == 0)  

        annotated_image = results[0].plot()
        cv2.imwrite(DETECTED_IMAGE_PATH, annotated_image)

        if person_count > 50:
            status = "unavailable (full)"
        elif 20 <= person_count <= 49:
            status = "available (crowded)"
        elif 1 <= person_count < 20:
            status = "available (clear)"
        else:
            status = "available (empty)"

        return jsonify({
            "people_count": person_count,
            "temperature": temperature,
            "weather_condition": condition,
            "status": status,
            "saved_image_path": DETECTED_IMAGE_PATH
        })

    except Exception as e:
        return jsonify({"error": str(e)})


def get_weather_data(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code != 200:
            return None

        data = response.json()
        temperature = data["main"]["temp"] 
        weather_condition = data["weather"][0]["description"]
        return {"temp": temperature, "condition": weather_condition}
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

