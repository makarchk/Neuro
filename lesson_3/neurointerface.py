import time
import threading
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import socket

from CapsuleSDK.Capsule import Capsule
from CapsuleSDK.DeviceLocator import DeviceLocator
from CapsuleSDK.DeviceType import DeviceType
from CapsuleSDK.Device import Device
from CapsuleSDK.EEGTimedData import EEGTimedData
from CapsuleSDK.Resistances import Resistances

from eeg_utils import *

# Конфиг
PLATFORM = 'mac'
EEG_WINDOW_SECONDS = 4.0
CHANNELS = 4
BUFFER_LEN = int(SAMPLE_RATE * EEG_WINDOW_SECONDS)
TARGET_SERIAL = None

# Настройки машинки и порога
ESP32_IP = "192.168.0.61"  # замените на IP вашей ESP32
UDP_PORT = 9999  # должен совпадать с main.py на ESP32
THRESHOLD = 5e-12  # порог мощности (подстройте под данные)
CALIBRATION_DURATION = 10.0  # секунд "тишины" при старте

# Ритмы
ALPHA_LOW, ALPHA_HIGH = 8.0, 12.0
BETA_LOW, BETA_HIGH = 13.0, 30.0
USE_BETA = False  # Переключите в True для задания 7.2

# UDP-сокет для управления
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_cmd(cmd):
    try:
        udp_sock.sendto((cmd + "\n").encode(), (ESP32_IP, UDP_PORT))
    except:
        pass


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

ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
channel_names = []
resistances_values = [0.0] * CHANNELS


def non_blocking_cond_wait(wake_event: EventFiredState, name: str, total_sleep_time: int):
    print(f"Waiting {name} up to {total_sleep_time}s...")
    steps = int(total_sleep_time * 50)
    for _ in range(steps):
        if device_locator is not None:
            try:
                device_locator.update()
            except:
                pass
        if wake_event.is_awake(): return True
        time.sleep(0.02)
    return False


def on_device_list(locator, info, fail_reason):
    global device
    chosen = None
    if len(info) == 0: return
    if TARGET_SERIAL is None:
        chosen = info[0]
    else:
        for dev in info:
            if dev.get_serial() == TARGET_SERIAL:
                chosen = dev
                break
    if chosen:
        print(f"Connecting to: {chosen.get_serial()}")
        device = Device(locator, chosen.get_serial(), locator.get_lib())
        device_list_event.set_awake()


def on_connection_status_changed(d, status):
    global channel_names
    ch_obj = device.get_channel_names()
    channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
    device_conn_event.set_awake()


def on_resistances(resistances_obj: Resistances):
    global resistances_values
    resistances_values = [resistances_obj.get_value(i) / 1000 for i in range(len(resistances_obj))]


def on_eeg(d, eeg: EEGTimedData):
    global ring
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()
    if samples <= 0: return
    block = np.zeros((ch, samples), dtype=float)
    for i in range(samples):
        for c in range(ch):
            block[c, i] = eeg.get_processed_value(c, i)
    ring.append_block(block[:CHANNELS, :])
    if not device_eeg_event.is_awake(): device_eeg_event.set_awake()


# Визуализация
fig, (ax_eeg, ax_psd) = plt.subplots(2, 1, figsize=(10, 8))
lines_eeg = [ax_eeg.plot([], [], lw=1)[0] for _ in range(CHANNELS)]
ax_eeg.set_title("EEG")
ax_eeg.grid(True)

lines_psd = [ax_psd.plot([], [], lw=1)[0] for _ in range(CHANNELS)]
thr_line = ax_psd.axhline(THRESHOLD, color='red', linestyle='--', label='Threshold')
ax_psd.set_title("PSD & Control Threshold")
ax_psd.set_xlim(0, 45)
ax_psd.set_ylim(0, 1e-11)
ax_psd.grid(True)
ax_psd.legend()

txt_status = fig.text(0.5, 0.02, "Status: Calibrating...", ha='center', fontweight='bold')
txt_imp = fig.text(0.02, 0.02, "Imp: ...", fontsize=8, family='monospace')

start_time = time.time()


def update_plot(_):
    global channel_names
    buf = ring.get()
    if buf.shape[1] == 0: return lines_eeg + lines_psd

    t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
    for i in range(CHANNELS):
        lines_eeg[i].set_data(t, buf[i, :])

    all_eeg = buf.flatten()
    ax_eeg.set_ylim(all_eeg.min() * 1.1, all_eeg.max() * 1.1)
    ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)

    try:
        freqs, psd = compute_psd_mne(buf, sfreq=SAMPLE_RATE, fmin=1.0, fmax=50.0)
        for i in range(min(psd.shape[0], CHANNELS)):
            lines_psd[i].set_data(freqs, psd[i, :])

        # Логика управления
        low, high = (BETA_LOW, BETA_HIGH) if USE_BETA else (ALPHA_LOW, ALPHA_HIGH)
        pows = integrate_band(freqs, psd, low, high)
        avg_pow = np.mean(pows)

        if (time.time() - start_time) > CALIBRATION_DURATION:
            if avg_pow > THRESHOLD:
                txt_status.set_text(f"FORWARD ({'BETA' if USE_BETA else 'ALPHA'} > Thr)")
                txt_status.set_color('green')
                send_cmd("F,50")
            else:
                txt_status.set_text("STOP / BACKWARD")
                txt_status.set_color('red')
                send_cmd("S")
        else:
            txt_status.set_text(f"Calibrating... {int(CALIBRATION_DURATION - (time.time() - start_time))}s")

    except:
        pass

    imp_str = "Imp (kOhm): " + " | ".join(
        [f"{channel_names[i] if i < len(channel_names) else i}: {resistances_values[i]:.1f}" for i in range(CHANNELS)])
    txt_imp.set_text(imp_str)

    return lines_eeg + lines_psd


def main():
    global device_locator, device
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, 'CapsuleClient.dll' if PLATFORM == 'win' else 'libCapsuleClient.dylib')

    capsuleLib = Capsule(lib_path)
    device_locator = DeviceLocator(capsuleLib.get_lib())
    device_locator.set_on_devices_list(on_device_list)
    device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)

    if not non_blocking_cond_wait(device_list_event, 'device list', 12): return

    device.set_on_connection_status_changed(on_connection_status_changed)
    device.set_on_eeg(on_eeg)
    device.set_on_resistances(lambda d, r: on_resistances(r))
    device.connect(bipolarChannels=False)

    if not non_blocking_cond_wait(device_conn_event, 'device connection', 20): return

    device.start()
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    running = True

    def updater():
        while running:
            try:
                device_locator.update()
            except:
                pass
            time.sleep(0.01)

    threading.Thread(target=updater, daemon=True).start()
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    running = False
    device.stop()
    device.disconnect()


if __name__ == '__main__':
    main()


def send_to_esp32(cmd):
    """Отправляет команду на ESP32."""
    try:
        udp_sock.sendto((cmd + "\n").encode(), (ESP32_IP, UDP_PORT))
    except OSError as e:
        print(f"[UDP] {e}")


# Инициализация
device = None
device_locator = None
calibration_start_time = None
is_calibrated = False
current_direction = "S"  # "F", "B", "S"


class EventFiredState:
    def __init__(self): self._awake = False

    def is_awake(self): return self._awake

    def set_awake(self): self._awake = True

    def sleep(self): self._awake = False


device_list_event = EventFiredState()
device_conn_event = EventFiredState()
device_eeg_event = EventFiredState()

ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
channel_names = []


def send_car_command(cmd):
    """Отправляет команду на ESP32."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        full_cmd = cmd + "\n"
        sock.sendto(full_cmd.encode('utf-8'), (ESP32_IP, UDP_PORT))
    except OSError as e:
        print(f"[CAR] Ошибка: {e}")
    finally:
        sock.close()


def non_blocking_cond_wait(wake_event, name, total_sleep_time):
    print(f"Waiting {name} up to {total_sleep_time}s...")
    steps = int(total_sleep_time * 50)
    for _ in range(steps):
        if device_locator is not None:
            try:
                device_locator.update()
            except:
                pass
        if wake_event.is_awake():
            return True
        time.sleep(0.02)
    return False


def on_device_list(locator, info, fail_reason):
    global device
    chosen = None
    if not info:
        print("No devices found.")
        return
    print(f"Found {len(info)} devices.")
    if TARGET_SERIAL is None:
        chosen = info[0]
    else:
        for dev in info:
            if dev.get_serial() == TARGET_SERIAL:
                chosen = dev
                break
    if not chosen:
        print(f"Target device {TARGET_SERIAL} not found!")
        return
    print(f"Connecting to: {chosen.get_name()} ({chosen.get_serial()})")
    device = Device(locator, chosen.get_serial(), locator.get_lib())
    device_list_event.set_awake()


def on_connection_status_changed(d, status):
    global channel_names
    ch_obj = device.get_channel_names()
    channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
    print(f"Channel names: {channel_names}")
    device_conn_event.set_awake()


rt_filter = RealTimeFilter(sfreq=250, l_freq=7, h_freq=13, n_channels=CHANNELS)


def on_eeg(d, eeg: EEGTimedData):
    global ring
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()
    if samples <= 0: return

    block = np.zeros((ch, samples), dtype=float)
    for i in range(samples):
        for c in range(ch):
            block[c, i] = eeg.get_processed_value(c, i)
    filtered_block = rt_filter.filter_block(block)

    if filtered_block.shape[0] >= CHANNELS:
        ring.append_block(filtered_block[:CHANNELS, :])
    else:
        padded = np.zeros((CHANNELS, filtered_block.shape[1]), dtype=float)
        padded[:filtered_block.shape[0], :] = filtered_block
        ring.append_block(padded)
    if not device_eeg_event.is_awake():
        device_eeg_event.set_awake()


#  Графики: добавляем порог на PSD
fig, (ax_eeg, ax_psd) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

lines_eeg = []
for i in range(CHANNELS):
    ln, = ax_eeg.plot([], [], label=f'Ch{i}', lw=1)
    lines_eeg.append(ln)
ax_eeg.set_ylabel("Amplitude (µV)")
ax_eeg.set_title("EEG Channels")
ax_eeg.legend(loc='upper right')
ax_eeg.grid(True)

lines_psd = []
for i in range(CHANNELS):
    ln, = ax_psd.plot([], [], label=f'PSD Ch{i}', lw=1)
    lines_psd.append(ln)
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("PSD (µV²/Hz)")
ax_psd.set_title("PSD Channels")
ax_psd.legend(loc='upper right')
ax_psd.grid(True)
ax_psd.set_xlim(0, 40)
ax_psd.set_ylim(0, 1e-10)

#  Серая пунктирная линия порога на PSD
thr_line = ax_psd.axhline(THRESHOLD, color='gray', linestyle='--', linewidth=1, label=f'Threshold = {THRESHOLD:.1e}')
ax_psd.legend(loc='upper right')


def update_plot(_):
    global channel_names, calibration_start_time, is_calibrated, current_direction

    try:
        device_locator.update()
    except Exception as e:
        print(f"[SDK] Update error: {e}")

    buf = ring.get()
    current_time = time.time()

    if buf.shape[1] == 0:
        return lines_eeg + lines_psd

    # Инициализация времени калибровки
    if calibration_start_time is None:
        calibration_start_time = current_time

    # Обновление EEG
    t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
    for i in range(CHANNELS):
        lines_eeg[i].set_data(t, buf[i, :])
        label = channel_names[i] if i < len(channel_names) else f'Ch{i}'
        lines_eeg[i].set_label(label)
    all_eeg = buf.flatten()
    ymin, ymax = all_eeg.min(), all_eeg.max()
    if ymin == ymax:
        ymin -= 1e-6;
        ymax += 1e-6
    pad = 0.1 * (ymax - ymin)
    ax_eeg.set_ylim(ymin - pad, ymax + pad)
    ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)
    ax_eeg.legend(loc='upper right')

    # Обновление PSD
    try:
        freqs, psd = compute_psd_mne(buf, sfreq=SAMPLE_RATE, fmin=1.0, fmax=50.0)
        num_ch = min(psd.shape[0], CHANNELS)
        for i in range(num_ch):
            lines_psd[i].set_data(freqs, psd[i, :])
            label = channel_names[i] if i < len(channel_names) else f'PSD Ch{i}'
            lines_psd[i].set_label(label)
        ax_psd.legend(loc='upper right')
    except Exception as e:
        print(f"[PSD] Error: {e}")
        return lines_eeg + lines_psd

    # Управление машинкой
    elapsed = current_time - calibration_start_time
    if not is_calibrated and elapsed >= CALIBRATION_DURATION:
        is_calibrated = True
        print("Калибровка завершена. Машинка готова к управлению.")

    if is_calibrated:
        # Решение: вперёд или назад
        if not (psd > THRESHOLD).any():
            if current_direction != "F":
                send_to_esp32("F,50")
                current_direction = "F"
        else:
            if current_direction != "B":
                send_to_esp32("B,50")
                current_direction = "B"

    return lines_eeg + lines_psd


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
    device.connect(bipolarChannels=True)
    if not non_blocking_cond_wait(device_conn_event, 'device connection', 40):
        print("Failed to connect.")
        return

    device.start()
    print("Device started. Opening plot...")

    # Начальная остановка машинки
    send_to_esp32("S")

    # интерактивный режим
    plt.ion()

    # Создаём анимацию без фонового потока
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    # Отображаем окно и не блокируем основной поток
    plt.show()

    # Основной цикл: ждём, пока окно не закроется
    try:
        while plt.fignum_exists(fig.number):  # пока окно открыто
            plt.pause(0.1)  # короткая пауза, чтобы не грузить CPU
    except KeyboardInterrupt:
        pass
    finally:
        send_to_esp32("S")
        udp_sock.close()
        device.stop()
        device.disconnect()
        print("Stopped.")


if __name__ == '__main__':
    main()