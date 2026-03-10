import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
img = cv2.imread("C:/Users/paulrithish-intern/Downloads/person.jpg")
resized_output = cv2.resize(img, (800, 600))

if img is None:
    print("Image not loaded")
else:
    # Run detection
    results = model(resized_output)
    # Loop through detected boxes
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Detect only person
        if model.names[cls_id] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(resized_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                resized_output,
                f"person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    cv2.imshow("Person Detection", resized_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()