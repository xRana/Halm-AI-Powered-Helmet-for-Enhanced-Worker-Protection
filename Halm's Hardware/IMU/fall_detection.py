import time
import board
import busio
import math
import requests
import json
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER
from adafruit_bno08x.i2c import BNO08X_I2C

# Constants for detection logic
FALL_THRESHOLD = 4.0
IMPACT_THRESHOLD = 45.0
FREEFALL_MIN_TIME = 0.1
I2C_ADDRESS = 0x4b
last_alert_time = 0
ALERT_COOLDOWN = 5

def notify_dashboard(alert_type, value):
    """Sends incident metadata to the cloud dashboard."""
    url = "https://Halm.pythonanywhere.com/api/hardware/alert"
    try:
        with open('current_location.json', 'r') as f:
            location = json.load(f)
        data = {
            "type": "Fall Detected",
            "impact_force": round(float(value), 2),
            "lat": location['lat'],
            "lon": location['lon']
        }
    except Exception as e:
        print(f"Location access error: {e}")
        return
    try:
        response = requests.post(url, json=data, timeout=2)
        print(f"Alert transmitted. Status: {response.status_code}")
    except Exception as e:
        print(f"Transmission error: {e}")

def setup_sensor():
    """Initializes the BNO085 IMU over I2C."""
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        bno = BNO08X_I2C(i2c, address=I2C_ADDRESS)
        bno.enable_feature(BNO_REPORT_ACCELEROMETER)
        return bno
    except Exception as e:
        print(f"Sensor initialization failed: {e}")
        return None

def get_magnitude(sensor):
    """Calculates the Euclidean norm of tri-axial acceleration."""
    ax, ay, az = sensor.acceleration
    return math.sqrt(ax**2 + ay**2 + az**2)

def monitor_helmet():
    """Main loop for kinematic monitoring and impact validation."""
    global last_alert_time
    sensor = setup_sensor()
    if not sensor:
        return
    
    print("Kinematic monitoring active...")
    while True:
        try:
            mag = get_magnitude(sensor)
            # Step 1: Detect Weightlessness (Free Fall)
            if mag < FALL_THRESHOLD:
                start = time.time()
                confirmed = False
                while mag < FALL_THRESHOLD:
                    mag = get_magnitude(sensor)
                    if (time.time() - start) > FREEFALL_MIN_TIME:
                        confirmed = True
                        break
                
                # Step 2: Validate Post-Fall Impact
                if confirmed:
                    t0 = time.time()
                    while (time.time() - t0) < 1.0:
                        impact = get_magnitude(sensor)
                        if impact > IMPACT_THRESHOLD:
                            now = time.time()
                            if (now - last_alert_time) > ALERT_COOLDOWN:
                                notify_dashboard("Kinematic Alert", impact)
                                last_alert_time = now
                            time.sleep(5) # Stabilization period
                            break
            time.sleep(0.01)
        except Exception as e:
            print(f"Runtime sensor error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    try:
        monitor_helmet()
    except KeyboardInterrupt:
        print("System stopped by user.")