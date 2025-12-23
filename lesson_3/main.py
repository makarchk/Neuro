import network
import socket
import time
from MX1508 import MX1508

# Настройки Wi-Fi
ssid = "WiFi"
password = "PASSWORD"

# Настройки UDP
UDP_PORT = 9999
WATCHDOG_TIMEOUT = 10

# Инициализация моторов
motor_L = MX1508(2, 4)
motor_R = MX1508(19, 18)

def pct_to_duty(percent):
    percent = max(0, min(100, int(percent)))
    return int(1023 * percent / 100)

def stop():
    motor_L.stop()
    motor_R.stop()

def forward_pct(percent):
    duty = pct_to_duty(percent)
    motor_L.forward(duty)
    motor_R.forward(duty)

def backward_pct(percent):
    duty = pct_to_duty(percent)
    motor_L.reverse(duty)
    motor_R.reverse(duty)

# Подключение к Wi-Fi
sta = network.WLAN(network.STA_IF)
sta.active(True)

if not sta.isconnected():
    print(f"Connecting to Wi-Fi '{ssid}'...")
    sta.connect(ssid, password)
    start_time = time.time()
    while not sta.isconnected() and (time.time() - start_time) < 15:
        time.sleep(0.2)

if sta.isconnected():
    print("Wi-Fi: connected")
    print("IP config:", sta.ifconfig())
else:
    print("Wi-Fi: FAILED")

stop()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.settimeout(0.1)
print(f"UDP server listening on port {UDP_PORT}...")

last_command_time = time.time()

try:
    while True:
        try:
            data, addr = sock.recvfrom(128)
            last_command_time = time.time()
            msg = data.decode().strip()
            if not msg: continue

            cmd = msg[0].upper()
            speed_pct = None
            if "," in msg:
                try: speed_pct = int(msg.split(",")[1])
                except: pass

            if cmd == "F":
                forward_pct(speed_pct if speed_pct is not None else 70)
                sock.sendto(b"ACK\n", addr)
            elif cmd == "B":
                backward_pct(speed_pct if speed_pct is not None else 60)
                sock.sendto(b"ACK\n", addr)
            elif cmd == "S":
                stop()
                sock.sendto(b"ACK\n", addr)
            elif cmd == "T":
                sock.sendto(b"OK\n", addr)
        except OSError:
            pass

        if time.time() - last_command_time > WATCHDOG_TIMEOUT:
            stop()
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    stop()
    try: sock.close()
    except: pass
    print("Bye")