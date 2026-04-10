
from ultralytics import YOLO
import cv2

# 1. Load the pre-trained YOLOv8 Nano model (it will auto-download on first run)
model = YOLO("yolov8n.pt")

# 2. Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)


# Check if webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Detecting... Press 'q' to stop.")

while True:
    # 3. Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Run the YOLO detection on the frame
    results = model(frame)

    # 5. Extract the frame with bounding boxes and labels drawn on it
    annotated_frame = results[0].plot()

    # Draw a filled rectangle (header background)
    cv2.rectangle(annotated_frame, (0, 0), (640, 60), (0, 0, 0), -1)

    # Then add text on top
    cv2.putText(
        annotated_frame,
        "Press 'Q' to Exit the program.",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    # 6. Show the frame in a window
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # 7. Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up
cap.release()
cv2.destroyAllWindows()
