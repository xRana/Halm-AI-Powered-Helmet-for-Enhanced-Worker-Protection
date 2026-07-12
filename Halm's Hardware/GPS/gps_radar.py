import socket
import json
import time

def start_radar_bridge():
    """Bridges GPSD satellite data to local system storage."""
    print("=========================================")
    print("      HALM SYSTEM: SATELLITE RADAR       ")
    print("=========================================")
    
    try:
        # Connect to the local GPSD daemon
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 2947))
        
        # Enable JSON streaming mode
        s.send(b'?WATCH={"enable":true,"json":true}\n')
        print("Connected to GPSD. Scanning for satellite fix...\n")
        
        buffer = ""
        last_print = 0
        
        while True:
            data = s.recv(4096).decode('utf-8')
            buffer += data
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if not line: continue
                
                try:
                    msg = json.loads(line)
                    
                    # Log satellite status (SKY Class)
                    if msg.get('class') == 'SKY':
                        satellites = msg.get('satellites', [])
                        visible_sats = len(satellites)
                        locked_sats = sum(1 for sat in satellites if sat.get('used', False))
                        
                        now = time.time()
                        if now - last_print > 1:
                            if locked_sats >= 3:
                                print(f"📡 Status: Locked ({locked_sats} satellites) - Strong Signal")
                            else:
                                print(f"⌛ Searching: {locked_sats} satellites locked. (Fix requires 3+)")
                            last_print = now
                            
                    # Extract coordinates (TPV Class)
                    elif msg.get('class') == 'TPV':
                        lat = msg.get('lat')
                        lon = msg.get('lon')
                        mode = msg.get('mode', 0)
                        
                        if mode >= 2 and lat and lon:
                            print(f"🌟 Fix Acquired -> Lat: {round(lat, 6)} | Lon: {round(lon, 6)}")
                            
                            location_data = {
                                "lat": lat,
                                "lon": lon,
                                "timestamp": time.time()
                            }
                            with open('current_location.json', 'w') as f:
                                json.dump(location_data, f)
                                
                except json.JSONDecodeError:
                    pass
            
    except ConnectionRefusedError:
        print("Error: GPSD service is not active. Please ensure the daemon is running.")
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Unexpected system error: {e}")

if __name__ == '__main__':
    start_radar_bridge()