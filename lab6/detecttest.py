from flask import Flask, jsonify
from flask_cors import CORS  # 用于支持跨域请求
from ultralytics import YOLO
import cv2
import requests
from datetime import datetime
import pytz
import threading
import time

app = Flask(__name__)
CORS(app)  # 启用 CORS 支持

# 配置路径和模式
SIMULATED_IMAGE_PATH = "/home/cs473/pi/picar-4wd/cs437-final/lab6/football.jpeg"
CAMERA_DETECT_PATH = "/home/cs473/pi/picar-4wd/cs437-final/lab6/cameradetect.png"
DETECTION_MODE = "camera"  # 默认使用摄像头模式

# OpenWeatherMap API 配置
API_KEY = "9f18703d01ed26a41982b6e32f5803c4"
LATITUDE = 40.1164  # Champaign, IL 的纬度
LONGITUDE = -88.2434  # Champaign, IL 的经度

# 状态变量
current_time = None
weather_data = {"temp": "unknown", "condition": "unknown"}
person_count = 0
status = "unknown"
latest_frame = None  # 存储最新捕获的摄像头帧
stop_camera = False  # 控制摄像头捕获线程停止

# 定时任务
def update_time():
    global current_time
    while True:
        chicago_tz = pytz.timezone("America/Chicago")
        current_time = datetime.now(chicago_tz).strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(1)  # 每秒更新一次时间

def update_weather():
    global weather_data
    while True:
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}&units=metric"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                weather_data["temp"] = data["main"]["temp"]
                weather_data["condition"] = data["weather"][0]["description"]
            else:
                print("Failed to fetch weather data.")
        except Exception as e:
            print(f"Error fetching weather data: {e}")
        time.sleep(1800)  # 每 30 分钟更新一次天气

def capture_camera():
    """
    实时捕获摄像头帧
    """
    global latest_frame, stop_camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while not stop_camera:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame  # 更新最新帧
        else:
            print("Error: Failed to capture frame.")

    cap.release()

def detect_people():
    """
    每 5 秒检测一次人数
    """
    global person_count, status
    model = YOLO("yolov8n.pt")  # 加载 YOLO 模型
    while True:
        try:
            if DETECTION_MODE == "image":
                image = cv2.imread(SIMULATED_IMAGE_PATH)
                if image is None:
                    print("Error: Failed to load image.")
                    time.sleep(5)
                    continue
            elif DETECTION_MODE == "camera":
                if latest_frame is None:
                    print("Warning: No frame captured yet.")
                    time.sleep(5)
                    continue
                image = latest_frame  # 使用最新帧

            # 使用 YOLO 模型检测人数
            results = model(image)
            person_count = sum(1 for obj in results[0].boxes.data if obj[5] == 0)

            # 保存检测结果图片（摄像头模式）
            if DETECTION_MODE == "camera":
                annotated_image = results[0].plot()
                cv2.imwrite(CAMERA_DETECT_PATH, annotated_image)

            # 判断状态
            if person_count > 50:
                status = "unavailable (full)"
            elif 20 <= person_count <= 49:
                status = "available (crowded)"
            elif 1 <= person_count < 20:
                status = "available (clear)"
            else:
                status = "available (empty)"

        except Exception as e:
            print(f"Error in detect_people: {e}")

        time.sleep(5)  # 每 5 秒检测一次

@app.route("/detect_status", methods=["GET"])
def detect_status():
    return jsonify({
        "current_time": current_time,
        "status": status,
        "temperature": f"{weather_data['temp']} °C",
        "weather_condition": weather_data["condition"],
        "people_count": person_count
    })

if __name__ == "__main__":
    try:
        # 启动定时任务
        threading.Thread(target=update_time, daemon=True).start()
        threading.Thread(target=update_weather, daemon=True).start()
        threading.Thread(target=capture_camera, daemon=True).start()
        threading.Thread(target=detect_people, daemon=True).start()

        # 启动 Flask 服务
        app.run(host="0.0.0.0", port=5000)

    except KeyboardInterrupt:
        stop_camera = True
        print("Shutting down server...")
