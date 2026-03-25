import torch
import cv2
import numpy as np

# Load YOLOv5 model (pre-trained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Starting Real-Time Object Detection... Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame (BGR to RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(rgb_frame)

    # Render results (bounding boxes)
    output_frame = np.squeeze(results.render())

    # Display output
    cv2.imshow("YOLOv5 Object Detection", output_frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
