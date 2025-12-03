# Курс "Нейронауки и нейроинтерфейсы" – Неделя 13

## 🧠 Проектное занятие №2

Добро пожаловать на второе проектное занятие по курсу Центрального Университета "Нейронауки и нейроинтерфейсы" в 2025 году! Сегодня мы применим библиотеку MNE-Python для анализа данных с нейроинтерфейса, визуализируем ЭЭГ и PSD всех каналов в реальном времени, рассчитаем мощность альфа-бета ритмов и напишем простой классификатор, чтобы в будущем отпралять 0/1 на какое-либо устройство (например, машинка)

---

### Цель занятия

Научиться визуализировать ЭЭГ, PSD в реальном времени с помощью MNE-Python, а также разработать простой классификатор на основе мощности альфа и бета ритмов

---

### По итогам занятия вы сможете:

1. Отрисовывать графики EEG, PSD и альфа-мощности в реальном времени
2. Использовать MNE-Python для анализа потоковых данных ЭЭГ
3. Реализовывать простой классификатор на основе порога альфа-мощности
4. Реализовывать свою собственную усложненную версию классификатора для управления устройством 

---

## Требования перед стартом

- Python 3.11+
- `CapsuleClient.dll` (Windows) или `libCapsuleClient.dylib` (macOS) в папке с кодом
- Подключенное устройство Neiry HeadBand по Bluetooth (в моменте сопряжения индикатор на Neiry Headband должен быстро мигать зеленым цветом)
- Установленные зависимости:
```bash
pip install numpy matplotlib mne
```

### Введение в eeg_utils.py

Файл `eeg_utils.py` — это набор вспомогательных функций и классов, предназначенных для обработки ЭЭГ-данных в реальном времени с использованием популярной библиотеки `MNE-Python`. Основная цель — упростить работу с потоковыми данными: буферизацию, преобразование в формат MNE, вычисление спектральной плотности мощности (PSD) и интеграцию мощности в заданном частотном диапазоне.

`RingBuffer`: Класс кольцевого буфера для хранения последних N сэмплов данных с нескольких каналов. Полезен для визуализации и анализа текущего "окна" данных.
`RealTimeFilter`: Класс для фильтрации ЭЭГ данных в заданном диапазоне.
`compute_psd_mne`: Функция для вычисления спектральной плотности мощности (PSD) с использованием MNE.
`integrate_band`: Функция для вычисления интегральной мощности в заданной полосе частот из данных PSD.
`SAMPLE_RATE`: Константа, определяющая частоту дискретизации (для Neiry HeadBand – 250 Гц).

### 1. Отрисовка ЭЭГ для всех каналов в реальном времени
---
Используется `matplotlib` и `FuncAnimation` для создания прокручивающегося графика ЭЭГ.

Данные ЭЭГ накапливаются в `RingBuffer`. `update_plot` функция вызывается регулярно, извлекает данные из буфера и обновляет линии графика.

#### Импорты библиотек, инициализации переменных
```python
import time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # PyQt5 via pip install PyQt5
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
```
#### Необходимые функции с прошлого занятия
```python
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
```
#### Функция обработки ЭЭГ сигнала с NeiryHeadBand
```python
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
```
#### Визуализация
```python
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
```
#### Основная функция подключения и получения информации
```python
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
```
--- 
### Задание 1.1

Используя `RealTimeFilter(sfreq, l_freq, h_freq, n_channels)` отфильтруйте данные ЭЭГ в диапазоне [1, 40], [7, 12], [13, 30] и посмотрите, что получится на графиках ЭЭГ

--- 

### 2. Отрисовка PSD для всех каналов в реальном времени
---

#### Визуализация
```python
# Отрисовка ЭЭГ и PSD на двух графиках
fig, (ax_eeg, ax_psd) = plt.subplots(2, 1, figsize=(10, 8), sharex=False) # Создаём два подграфика

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
ax_psd.set_xlabel("Frequency (Hz)") # Ось X для PSD
ax_psd.set_ylabel("PSD (µV²/Hz)")
ax_psd.set_title("PSD Channels")
ax_psd.legend(loc='upper right')
ax_psd.grid(True)
```
#### Обновление функции `update_plot` для отображения PSD
```python
def update_plot(_):
    global channel_names

    # Обновление EEG
    buf = ring.get()
    if buf.shape[1] != 0:
        t = np.linspace(-EEG_WINDOW_SECONDS, 0, buf.shape[1])
        for i in range(CHANNELS):
            lines_eeg[i].set_data(t, buf[i, :])
            if len(channel_names) > i:
                lines_eeg[i].set_label(channel_names[i])
            else:
                lines_eeg[i].set_label(f'Ch{i}')

        all_data_eeg = buf.flatten()
        ymin, ymax = all_data_eeg.min(), all_data_eeg.max()
        if ymin == ymax:
            ymin -= 1e-6; ymax += 1e-6
        pad = 0.1*(ymax - ymin)
        ax_eeg.set_ylim(ymin-pad, ymax+pad)
        ax_eeg.set_xlim(-EEG_WINDOW_SECONDS, 0)
        ax_eeg.legend(loc='upper right')

        # Вычисление PSD из буфера
        freqs, psd = compute_psd_mne(
            buf, 
            sfreq=SAMPLE_RATE, 
            fmin=1.0, 
            fmax=50.0, 
            n_fft=int(SAMPLE_RATE * 2)  # 2-секундное окно
        )

        # Отрисовка PSD
        num_ch = min(psd.shape[0], CHANNELS)
        for i in range(num_ch):
            lines_psd[i].set_data(freqs, psd[i, :])
            lines_psd[i].set_label(f'{channel_names[i] if i < len(channel_names) else f"Ch{i}"}')

        # Масштаб PSD
        # all_psd = psd[:, freqs <= 40].flatten()
        # if all_psd.size > 0:
        #     ymin_psd, ymax_psd = all_psd.min(), all_psd.max()
        #     if ymin_psd == ymax_psd:
        #         ymin_psd -= 1e-12; ymax_psd += 1e-12
        #     pad_psd = 0.1 * (ymax_psd - ymin_psd)
        ax_psd.set_ylim(0, 1e-12)
        ax_psd.set_xlim(0, 40)
        ax_psd.legend(loc='upper right')

    return lines_eeg + lines_psd
```
--- 
### Задание 2.1

Используя `RealTimeFilter(sfreq, l_freq, h_freq, n_channels)` отфильтруйте данные ЭЭГ в диапазоне [1, 40], [7, 12], [13, 30] и посмотрите, что получится на графиках ЭЭГ и PSD

--- 

### 3. Отрисовка альфа мощности для всех каналов в реальном времени
---
#### Дополнительные переменные
```python
# Параметры альфа-диапазона
ALPHA_LOW = 8.0   # Гц
ALPHA_HIGH = 12.0 # Гц

# История для графика альфа-мощности
alpha_history = [[] for _ in range(CHANNELS)]
time_history = []
```
#### Создание подграфиков для ЭЭГ, PSD и альфа-мощности
```python
# Отрисовка ЭЭГ,PSD, alpha power на трех графиках
fig, (ax_eeg, ax_psd, ax_alpha) = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

lines_alpha = []
for i in range(CHANNELS):
    ln, = ax_alpha.plot([], [], label=f'Alpha Ch{i}', lw=1)
    lines_alpha.append(ln)
ax_alpha.set_xlabel("Time (s)") 
ax_alpha.set_ylabel("Alpha Power (µV²/Hz)")
ax_alpha.set_title("Alpha Power (8-12 Hz)")
ax_alpha.legend(loc='upper right')
ax_alpha.grid(True)
```
#### Обновление функции `update_plot` для отображения альфа-мощности
```python
def update_plot(_):
    global channel_names, alpha_history, time_history

    buf = ring.get()
    current_time = time.time()

    # Обновление EEG
    #...

        # Обновляем PSD-график
        #...

        # Масштаб PSD
        #...

        # Вычисление и обновление Alpha Power
        current_alpha = []
        for i in range(num_ch):
            alpha_pow = integrate_band(freqs, psd[i, :], 8, 13)
            current_alpha.append(alpha_pow)

        while len(current_alpha) < CHANNELS:
            current_alpha.append(0.0)

        # Обновляем историю
        time_history.append(current_time)
        for i in range(CHANNELS):
            alpha_history[i].append(current_alpha[i])

        MAX_HISTORY = 40
        if len(time_history) > MAX_HISTORY:
            time_history[:] = time_history[-MAX_HISTORY:]
            for i in range(CHANNELS):
                alpha_history[i] = alpha_history[i][-MAX_HISTORY:]

        # Относительное время для графика
        if time_history:
            t0 = time_history[0]
            t_rel = [t - t0 for t in time_history]
        else:
            t_rel = []

        # Обновляем линии альфа-мощности
        for i in range(CHANNELS):
            lines_alpha[i].set_data(t_rel, alpha_history[i])
            label = f'Alpha {channel_names[i] if i < len(channel_names) else f"Ch{i}"}'
            lines_alpha[i].set_label(label)

        # Масштабируем альфа-график
        if t_rel:
            ax_alpha.set_xlim(t_rel[0], t_rel[-1])
            # all_alpha = np.concatenate([np.array(h) for h in alpha_history if h])
            # if all_alpha.size > 0:
            #     ymin_a, ymax_a = all_alpha.min(), all_alpha.max()
            #     if ymin_a == ymax_a:
            #         ymin_a -= 1e-12; ymax_a += 1e-12
            #     pad_a = 0.1 * (ymax_a - ymin_a)
            ax_alpha.set_ylim(0, 1e-11)
            ax_alpha.legend(loc='upper right')

    return lines_eeg + lines_psd + lines_alpha
```
---
#### Задание 3.1
Попробуйте отрисовать мощность бета ритма

---
### 4. Базовый классификатор для всех каналов (через порог)
---
#### Инициализация переменных для порога альфа-мощности и подсчёта прогресса по каналам
```python
# Порог и настройки прогресса
ALPHA_THRESHOLD = 5e-12  # ← настройте под ваши данные
CALIBRATION_DURATION = 10.0  # секунд до старта прогресса
progress_value = 0  # 0..100
last_accum_time = None
progress_step_sec = 0.1  # шаг обновления прогресса

# Замените историю: теперь одна линия — среднее
alpha_avg_history = []  # вместо alpha_history = [[] for _ in range(CHANNELS)]
time_history = []
```
#### Создание подграфиков для ЭЭГ, PSD, альфа-мощности и прогресс-бара по каналам
```python
# Отрисовка ЭЭГ, PSD, Alpha (среднее), Progress
fig, (ax_eeg, ax_psd, ax_alpha, ax_progress) = plt.subplots(4, 1, figsize=(10, 12), sharex=False)
```
#### Создание подграфиков для ЭЭГ, PSD, альфа-мощности и прогресс-бара по каналам
```python
# Alpha: ОДНА линия — среднее по каналам
line_alpha_avg, = ax_alpha.plot([], [], 'b-', lw=2, label='Avg Alpha')
# Серая пунктирная линия порога
thr_line = ax_alpha.axhline(ALPHA_THRESHOLD, color='gray', linestyle='--', linewidth=1, label='Threshold')
ax_alpha.set_xlabel("Time (s)") 
ax_alpha.set_ylabel("Alpha Power (µV²/Hz)")
ax_alpha.set_title("Average Alpha Power (8–12 Hz)")
ax_alpha.set_ylim(0, 1e-11)
ax_alpha.legend(loc='upper right')
ax_alpha.grid(True)

# рогресс-бар (4-й график)
ax_progress.set_xlim(0, 100)
ax_progress.set_ylim(-0.5, 0.5)
ax_progress.set_xlabel("Progress (%)")
ax_progress.set_ylabel("")
ax_progress.set_title("Alpha > Threshold → +1 every 0.1s (after 10s)")
bar_container = ax_progress.barh([0], [0], height=0.8, color='tab:green', alpha=0.8)
progress_bar = bar_container[0]
ax_progress.set_yticks([])
ax_progress.grid(True, axis='x')
```
#### Обновление прогресса для каждого канала в функции `update_plot`
```python
global alpha_avg_history, time_history, progress_value, last_accum_time

# Средняя альфа по всем каналам
    current_alpha = []
    for i in range(num_ch):
        alpha_pow = integrate_band(freqs, psd[i, :], ALPHA_LOW, ALPHA_HIGH)
        current_alpha.append(alpha_pow)
    avg_alpha = np.mean(current_alpha) if current_alpha else 0.0

    time_history.append(current_time)
    alpha_avg_history.append(avg_alpha)

    MAX_HISTORY = 100
    if len(time_history) > MAX_HISTORY:
        time_history[:] = time_history[-MAX_HISTORY:]
        alpha_avg_history[:] = alpha_avg_history[:][-MAX_HISTORY:]

    # Обновляем график средней альфа
    if time_history:
        t0 = time_history[0]
        t_rel = [t - t0 for t in time_history]
        line_alpha_avg.set_data(t_rel, alpha_avg_history)
        ax_alpha.set_xlim(t_rel[0], t_rel[-1])

    # Прогресс-бар: +1/-1 каждые 0.1 сек, но только после 10 сек
    if len(time_history) > 0 and (current_time - time_history[0]) >= CALIBRATION_DURATION:
        now = time.time()
        if last_accum_time is None:
            last_accum_time = now
        if now - last_accum_time >= progress_step_sec:
            if avg_alpha > ALPHA_THRESHOLD:
                progress_value = min(100, progress_value + 1)
            else:
                progress_value = max(0, progress_value - 1)
            last_accum_time = now
        progress_bar.set_width(progress_value)
    else:
        progress_bar.set_width(0)  # до 10 сек — ноль

    return lines_eeg + lines_psd + [line_alpha_avg] + [progress_bar]
```
---

#### Задание 4.1

Попробуйте самостоятельно поподбирать пороги для классифкатора и поуправлять прогресс-баром

### Итоги

- Освоили визуализацию ЭЭГ и PSD в реальном времени с помощью `matplotlib`.
- Применили функции из MNE для вычисления мощности в определённых диапазонах (альфа, бета).
- Реализовали простой "классификатор" на основе порога альфа-мощности.

### Что будет на следующем занятии

- Знакомство с ESP32
- Управление машинкой путем отправки данных
- Интеграция Neiry HeadBand с ESP32

### Попробовать дома

1) Добавить отображение импеданса в графическом интерфейсе
2) Запрограммировать считывание мощностей альфа, бета ритмов
3) Реализовать алгоритм для управления устройством (алгоритм выдает 0/1 или go/no go)
4) Изучить заранее специфику работы ESP32
5) Выбрать оператора для управления устройством
