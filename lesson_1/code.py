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
from CapsuleSDK.Resistances import Resistances
from CapsuleSDK.PSDData import PSDData
from CapsuleSDK.Calibrator import Calibrator, IndividualNFBData

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lesson_2'))
from eeg_utils import RingBuffer, compute_psd_mne, SAMPLE_RATE

PLATFORM = 'mac'  # 'mac' or 'win'
CHANNELS = 4
EEG_WINDOW_SECONDS = 4.0
BUFFER_LEN = int(SAMPLE_RATE * EEG_WINDOW_SECONDS)
TARGET_SERIAL = None


class EventFiredState:
    def __init__(self): self._awake = False

    def is_awake(self): return self._awake

    def set_awake(self): self._awake = True

    def sleep(self): self._awake = False


device_locator = None
device = None
device_list_event = EventFiredState()
device_conn_event = EventFiredState()
device_eeg_event = EventFiredState()
calibrated_event = EventFiredState()

ring = RingBuffer(n_channels=CHANNELS, maxlen=BUFFER_LEN)
channel_names = []


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
    print("Name:", chosen.get_name())
    print("Type:", chosen.get_type())

    device = Device(locator, chosen.get_serial(), locator.get_lib())
    device_list_event.set_awake()


def on_connection_status_changed(dev, status):
    global channel_names
    print("Connection status changed:", status)
    ch_obj = device.get_channel_names()
    channel_names = [ch_obj.get_name_by_index(i) for i in range(len(ch_obj))]
    print(f"Channel names: {channel_names}")
    device_conn_event.set_awake()


def on_resistances(resistances_obj: Resistances):
    values = [resistances_obj.get_value(i) / 1000 for i in range(len(resistances_obj))]
    print("Resistances (kOhm):", [f"{v:.2f}" for v in values])


def on_eeg(dev, eeg: EEGTimedData):
    global ring
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()

    if samples > 0:
        # Задание 3: Получение сырых (raw) данных
        raw_vals = [eeg.get_raw_value(c, 0) for c in range(ch)]
        # print(" first raw sample:", raw_vals) # Раскомментируйте для отладки

        block = np.zeros((ch, samples), dtype=float)
        for i in range(samples):
            for c in range(ch):
                block[c, i] = eeg.get_processed_value(c, i)

        if block.shape[0] >= CHANNELS:
            ring.append_block(block[:CHANNELS, :])
        else:
            padded = np.zeros((CHANNELS, block.shape[1]), dtype=float)
            padded[:block.shape[0], :] = block
            ring.append_block(padded)

    if not device_eeg_event.is_awake():
        device_eeg_event.set_awake()


def on_calibrated(calibrator: Calibrator, data: IndividualNFBData):
    print("Calibration finished.")
    print(" timestamp:", data.timestampMilli)
    print(" lowerFrequency:", data.lowerFrequency)
    print(" upperFrequency:", data.upperFrequency)
    print(" individualFrequency:", data.individualFrequency)
    calibrated_event.set_awake()



fig, (ax_eeg, ax_psd) = plt.subplots(2, 1, figsize=(10, 8))

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


def update_plot(_):
    global channel_names
    buf = ring.get()
    if buf.shape[1] == 0:
        return lines_eeg + lines_psd

    t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
    for i in range(CHANNELS):
        lines_eeg[i].set_data(t, buf[i, :])
        label = channel_names[i] if i < len(channel_names) else f'Ch{i}'
        lines_eeg[i].set_label(label)

    all_eeg = buf.flatten()
    ymin, ymax = all_eeg.min(), all_eeg.max()
    if ymin == ymax: ymin -= 1e-6; ymax += 1e-6
    pad = 0.1 * (ymax - ymin)
    ax_eeg.set_ylim(ymin - pad, ymax + pad)
    ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)
    ax_eeg.legend(loc='upper right')

    try:
        freqs, psd = compute_psd_mne(buf, sfreq=SAMPLE_RATE, fmin=1.0, fmax=50.0)
        num_ch = min(psd.shape[0], CHANNELS)
        for i in range(num_ch):
            lines_psd[i].set_data(freqs, psd[i, :])
            label = channel_names[i] if i < len(channel_names) else f'PSD Ch{i}'
            lines_psd[i].set_label(label)
        ax_psd.legend(loc='upper right')
    except Exception as e:
        print(f"PSD Error: {e}")

    return lines_eeg + lines_psd


def main():
    global device_locator, device

    if PLATFORM == 'win':
        capsuleLib = Capsule('./CapsuleClient.dll')
    else:
        capsuleLib = Capsule('./libCapsuleClient.dylib')

    device_locator = DeviceLocator(capsuleLib.get_lib())
    device_locator.set_on_devices_list(on_device_list)

    # Для отладки без устройства можно использовать DeviceType.Noise
    device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)

    if not non_blocking_cond_wait(device_list_event, 'device list', 12):
        print("No device found. Exiting.")
        return

    device.set_on_connection_status_changed(on_connection_status_changed)
    device.set_on_eeg(on_eeg)
    device.set_on_resistances(lambda dev, res: on_resistances(res))

    device.connect(bipolarChannels=False)
    if not non_blocking_cond_wait(device_conn_event, 'device connection', 20):
        print("Failed to connect.")
        return

    device.start()

    try:
        info = device.get_info()
        print("Device info:")
        print("serial:", info.get_serial())
        print("name:", info.get_name())
        print("firmware:", info.get_firmware())
        print("type:", info.get_type())
    except Exception as e:
        print("Failed to get info:", e)

    print("Opening plot...")
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    running = True

    def updater():
        while running:
            try:
                device_locator.update()
            except:
                pass
            time.sleep(0.01)

    t = threading.Thread(target=updater, daemon=True)
    t.start()

    plt.tight_layout()
    plt.show()

    running = False
    device.stop()
    device.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()