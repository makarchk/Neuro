import time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from CapsuleSDK.Capsule import Capsule
from CapsuleSDK.DeviceLocator import DeviceLocator
from CapsuleSDK.DeviceType import DeviceType
from CapsuleSDK.Device import Device
from CapsuleSDK.EEGTimedData import EEGTimedData

from eeg_utils import *

# Конфиг
PLATFORM = 'mac'
EEG_WINDOW_SECONDS = 4.0 # Размер окна ЭЭГ данных
CHANNELS = 4 #4 если Bipolar=False
BUFFER_LEN = int(SAMPLE_RATE * EEG_WINDOW_SECONDS) # Размер буффера 
MAX_PLOT_CHANNELS = CHANNELS # Сколько каналов отрисовывать
TARGET_SERIAL = None # Серийний нейроинтерфейса для подключения, например "821619"

device = None
device_locator = None

class EventFiredState:
    def __init__(self): self._awake = False
    def is_awake(self): return self._awake
    def set_awake(self): self._awake = True
    def sleep(self): self._awake = False

device_list_event = EventFiredState()
device_conn_event = EventFiredState()
device_eeg_event = EventFiredState()

# Инициализация буфера (n_channels, buffer_len)
ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
# Инициализация channel_names как пустого списка, будет заполнено позже
channel_names = []

# Обработчик списка событий
def non_blocking_cond_wait(wake_event: EventFiredState, name: str, total_sleep_time: int):
    print(f"Waiting {name} up to {total_sleep_time}s...")
    steps = int(total_sleep_time * 50)
    for _ in range(steps):
        if device_locator is not None:
            try:
                device_locator.update()
            except Exception:
                pass
        if wake_event.is_awake():
            return True
        time.sleep(0.02)
    return False

# Обработчик получения списка найденных устройств
# Вызывается автоматически SDK после сканирования
def on_device_list(locator, info, fail_reason):
    global device
    chosen = None
    if len(info) == 0:
        print("No devices found.")
        return
    print(f"Found {len(info)} devices.")
    if TARGET_SERIAL is None:
        print(f"Using first device:")
        chosen = info[0]
    else:
        for dev in info:
            print(" device:", dev.get_serial(), dev.get_name())
            if dev.get_serial() == TARGET_SERIAL:
                chosen = dev
                break
    if chosen is None:
        print(f"Target device {TARGET_SERIAL} not found!")
        return
    print()
    print("Connecting to:")
    print("Serial:", chosen.get_serial())
    print("Name:  ", chosen.get_name())
    print("FW:    ", chosen.get_firmware())
    print("Type:  ", chosen.get_type())
    device = Device(locator, chosen.get_serial(), locator.get_lib())
    device_list_event.set_awake()

# Обработчик изменения статуса подключения устройства
# Вызывается автоматически SDK при изменении статуса
def on_connection_status_changed(d, status): 
    global channel_names
    print("Connection status changed:", status)
    ch_obj = device.get_channel_names()
    channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
    print(f"Channel names: {channel_names}")
    device_conn_event.set_awake()

def on_eeg(d, eeg: EEGTimedData):
    global ring
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()
    
    if samples <= 0: return

    block = np.zeros((ch, samples), dtype=float)
    for i in range(samples):
        for c in range(ch):
            block[c, i] = eeg.get_processed_value(c, i)
            # TO DO
            # TO DO

    if block.shape[0] >= CHANNELS:
        ring.append_block(block[:CHANNELS, :])
    else:
        padded = np.zeros((CHANNELS, block.shape[1]), dtype=float)
        padded[:block.shape[0], :] = block
        ring.append_block(padded)
    if not device_eeg_event.is_awake():
        device_eeg_event.set_awake()

# Отрисовка на одном графике
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
for i in range(CHANNELS):
    ln, = ax.plot([], [], label=f'Ch{i}', lw=1)
    lines.append(ln)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (µV)")
ax.set_title("EEG Channels")
ax.legend(loc='upper right')
ax.grid(True)

def update_plot(_):
    global channel_names
    buf = ring.get()
    if buf.shape[1] == 0:
        return lines
    t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
    # Обновляем данные
    for i in range(CHANNELS):
        lines[i].set_data(t, buf[i, :])
        # Обновляем имена каналов, если доступны
        if len(channel_names) > i:
             lines[i].set_label(channel_names[i])
        else:
             lines[i].set_label(f'Ch{i}')

    # Динамическое масштабирование по Y
    all_data = buf.flatten()
    ymin, ymax = all_data.min(), all_data.max()
    if ymin == ymax:
        ymin -= 1e-6; ymax += 1e-6
    pad = 0.1*(ymax - ymin)
    ax.set_ylim(ymin-pad, ymax+pad)
    ax.set_xlim(-EEG_WINDOW_SECONDS, 0)
    # Перерисовываем легенду
    ax.legend(loc='upper right')
    return lines

def main():
    global device_locator, device
    if PLATFORM == 'win':
        capsuleLib = Capsule('./CapsuleClient.dll')
    else:
        capsuleLib = Capsule('./libCapsuleClient.dylib')

    device_locator = DeviceLocator(capsuleLib.get_lib())
    device_locator.set_on_devices_list(on_device_list)
    device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)

    if not non_blocking_cond_wait(device_list_event, 'device list', 12):
        print("No device found. Exiting.")
        return

    device.set_on_connection_status_changed(on_connection_status_changed)
    device.set_on_eeg(on_eeg)
    device.connect(bipolarChannels=False)
    if not non_blocking_cond_wait(device_conn_event, 'device connection', 20):
        print("Failed to connect.")
        return

    device.start()
    print("Device started. Opening plot...")

    # Создаём анимацию matplotlib, которая будет обновлять график в реальном времени.
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    running = True
    # Фоновая функция, которая регулярно опрашивает устройство
    def updater():
        while running:
            try:
                device_locator.update()
            except Exception:
                pass
            time.sleep(0.01)

    # Создаём и запускаем фоновый поток (daemon=True означает, что поток завершится,
    # когда завершится основной поток — например, при закрытии окна графика)
    t = threading.Thread(target=updater, daemon=True)
    t.start()
    plt.tight_layout()
    plt.show()

    running = False
    device.stop()
    device.disconnect()
    print("Stopped.")

if __name__ == '__main__':
    main()
