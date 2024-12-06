from ultralytics import YOLO
import cv2

def yolo_realtime_detection():
    # 加载 YOLOv8n 模型
    model = YOLO("yolov8n.pt")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit the detection feed.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 使用 YOLOv8n 检测
        results = model(frame)

        # 在帧上绘制检测框
        annotated_frame = results[0].plot()

        # 显示检测结果
        cv2.imshow("YOLOv8n Realtime Detection", annotated_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo_realtime_detection()
