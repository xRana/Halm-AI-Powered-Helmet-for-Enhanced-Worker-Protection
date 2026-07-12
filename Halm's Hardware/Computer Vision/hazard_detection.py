import cv2
import onnxruntime as ort
import numpy as np
from picamera2 import Picamera2
from gpiozero import LED 
import time
import requests
import json

# Alert control variables
last_alert_time = 0
ALERT_COOLDOWN = 5 

def notify_dashboard(alert_type, confidence):
    """Transmits visual hazard metadata to the Safety Operations Center."""
    url = "https://Halm.pythonanywhere.com/api/hardware/alert"
    try:
        with open('current_location.json', 'r') as f:
            location = json.load(f)
            data = {
                "type": alert_type,      
                "confidence": round(float(confidence * 100), 2),
                "lat": location['lat'],
                "lon": location['lon']
            }
    except Exception as e:
        print(f"Location file read error: {e}")
        return

    try:
        response = requests.post(url, json=data, timeout=2) 
        print(f"Alert successfully transmitted. Status: {response.status_code}")
    except Exception as e:
        print(f"Network error: Alert transmission failed: {e}")
        
time.sleep(2)
print("Initializing Halm Vision Engine...")

# Model and Peripherals Setup
class_names = ['sharp_tool', 'Pit', 'Handheld_tool'] 
model_path = "/home/halm/Downloads/best.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
alert_led = LED(18)

# Camera Configuration
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

print("Inference engine active. Press 'q' to stop.")

try:
    while True:
        frame = picam2.capture_array()
        
        # Image Pre-processing for ONNX model
        img = cv2.resize(frame, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Inference Pass
        outputs = session.run(None, {session.get_inputs()[0].name: img})
        output = np.squeeze(outputs[0]).T
        
        boxes, confidences, class_ids = [], [], []
        hazard_detected = False
        detected_label, highest_conf = "", 0.0

        for row in output:
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = class_names[class_id] if class_id < len(class_names) else ""
                
                # Filter for environmental hazards
                if label in ['sharp_tool', 'Pit']:
                    hazard_detected = True
                    detected_label = "Sharp Tool" if label == 'sharp_tool' else "Pothole"
                    if confidence > highest_conf:
                        highest_conf = confidence

                xc, yc, w, h = row[:4]
                x1 = int((xc - w/2) * (frame.shape[1] / 640))
                y1 = int((yc - h/2) * (frame.shape[0] / 640))
                boxes.append([x1, y1, int(w * (frame.shape[1] / 640)), int(h * (frame.shape[0] / 640))])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Alert Logic and Local Feedback
        current_time = time.time()
        if hazard_detected:
            alert_led.on()  
            if (current_time - last_alert_time) > ALERT_COOLDOWN:
                print(f"Critical Event Detected: {detected_label}")
                notify_dashboard(detected_label, highest_conf)
                last_alert_time = current_time
        else:
            alert_led.off()

        # Non-Maximum Suppression and Drawing
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_names[class_ids[i]]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Real-Time Hazard Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    alert_led.off() 
    alert_led.close()
    print("Vision system terminated.")