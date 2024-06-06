import torch
import cv2
import numpy as np
import win32process
import os
import pygetwindow as gw
import psutil
import time
import keyboard
import threading
import win32gui
import win32api
import win32con
import mss
import tkinter as tk
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.text import Text
from rich.prompt import Prompt

CONFIDENCE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.5
WINDOW_TITLE = "Telegram"
STOP_SIGNAL = False
CTRL_Q_PRESSED_ONCE = False
FRAME_SKIP = 1
PAUSE_SIGNAL = False
SETTINGS_SIGNAL = False
TARGET_ID = 1

DELAY_BETWEEN_CLICKS = 0
DELAY_BEFORE_CLICK = 0

click_counters = {}

class MessagesPanel(Panel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_lines = 10
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)
        self.renderable = Text.from_markup("\n".join(self.messages[-self.max_lines:]))

class CustomConsole(Console):
    def __init__(self, *args, messages_panel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages_panel = messages_panel

    def log(self, *objects, **kwargs):
        highlight = kwargs.pop("highlight", False)
        message = Text.assemble(*objects)
        if highlight:
            message.stylize("bold")

        if self.messages_panel:
            formatted_message = f"[{time.strftime('%H:%M:%S')}] {message.markup}"
            self.messages_panel.add_message(formatted_message)
        else:
            super().log(message, **kwargs)

def load_model(console):
    console.log(Text("Loading model...", style="blue"))
    default_model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    model_path = Prompt.ask(Text("Path to model weights file", style="bold magenta"), default=default_model_path)
    with Progress() as progress:
        task = progress.add_task("[cyan]Loading...", total=100)
        device = select_device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DetectMultiBackend(model_path, device=device)
        model = model.to(device)
        model.warmup(imgsz=(1, 3, 416, 416))
        progress.update(task, advance=100)
    console.log(Text("Model loaded!", style="green"), highlight=True)
    return model, device

def preprocess_image(image, device):
    image = cv2.resize(image, (416, 416))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).float().div(255.0).unsqueeze(0).to(device)

def find_telegram_window(console, title_keyword=WINDOW_TITLE):
    console.log(Text(f"Searching for window '{title_keyword}'...", style="blue"), highlight=True)
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        console.log(Text(f"Window '{title_keyword}' not found", style="red"), highlight=True)
        return None
    console.log(Text(f"Window '{windows[0].title}' found!", style="green"), highlight=True)
    return windows[0]

def update_window_coordinates(window):
    window.update()
    return {"left": window.left, "top": window.top, "width": window.width, "height": window.height}

def capture_telegram_window(console, window):
    global STOP_SIGNAL, PAUSE_SIGNAL
    hwnd = window._hWnd
    while not STOP_SIGNAL:
        if PAUSE_SIGNAL or SETTINGS_SIGNAL:
            time.sleep(0.1)
            continue

        window_rect = win32gui.GetWindowRect(hwnd)
        bbox = {"left": window_rect[0], "top": window_rect[1], "width": window_rect[2] - window_rect[0], "height": window_rect[3] - window_rect[1]}

        with mss.mss() as sct:
            screenshot = sct.grab(bbox)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            yield img, bbox

def perform_click(console, x, y):
    global click_counters
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    click_key = (x, y)
    if click_key not in click_counters:
        click_counters[click_key] = 0
    click_counters[click_key] += 1

    return x, y, click_counters[click_key]

def check_keyboard(console):
    global STOP_SIGNAL, CTRL_Q_PRESSED_ONCE, PAUSE_SIGNAL, SETTINGS_SIGNAL
    while True:
        if keyboard.is_pressed("ctrl+q"):
            if not CTRL_Q_PRESSED_ONCE:
                console.log(Text("CTRL+Q pressed. Press again to exit.", style="yellow"), highlight=True)
                CTRL_Q_PRESSED_ONCE = True
            else:
                console.log(Text("Exiting...", style="green"), highlight=True)
                STOP_SIGNAL = True
                break
        elif keyboard.is_pressed("ctrl+x"):
            if not PAUSE_SIGNAL:
                console.log(Text("Script paused. Press CTRL+X again to resume.", style="magenta"), highlight=True)
                PAUSE_SIGNAL = True
            else:
                console.log(Text("Script resumed.", style="magenta"), highlight=True)
                PAUSE_SIGNAL = False
        elif keyboard.is_pressed("ctrl+w"):
            console.log(Text("Opening settings panel...", style="cyan"), highlight=True)
            SETTINGS_SIGNAL = True
        time.sleep(0.1)

def get_system_info():
    cpu_usage = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_usage = mem.percent
    mem_total = mem.total / (1024**3)
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            gpu_info = f"{gpu.name} - {gpu.total_memory // (1024 ** 2)} MB"
            gpu_usage = torch.cuda.memory_allocated(0) / gpu.total_memory * 100
            gpu_usage = f"{gpu_usage:.2f}%"
        else:
            gpu_info = "GPU not available"
            gpu_usage = "N/A"
    except Exception as e:
        gpu_info = f"GPU error: {type(e).__name__}: {e}"
        gpu_usage = "N/A"
    return {
        "CPU Usage": f"{cpu_usage}%",
        "Memory Usage": f"{mem_usage}%",
        "Total Memory": f"{mem_total:.2f} GB",
        "GPU": gpu_info,
        "GPU Usage": gpu_usage,
    }

def bring_window_to_foreground(console, window):
    hwnd = window._hWnd
    try:
        window_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
        current_thread = win32api.GetCurrentThreadId()

        if window_thread != current_thread:
            win32process.AttachThreadInput(window_thread, current_thread, True)

        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetFocus(hwnd)

        if window_thread != current_thread:
            win32process.AttachThreadInput(window_thread, current_thread, False)

        console.log(Text(f"Window '{window.title}' brought to the foreground!", style="green"), highlight=True)
    except Exception as e:
        console.log(Text(f"Error: {e}", style="red"), highlight=True)

def show_settings(console):
    global DELAY_BETWEEN_CLICKS, DELAY_BEFORE_CLICK, SETTINGS_SIGNAL

    def save_settings():
        global DELAY_BETWEEN_CLICKS, DELAY_BEFORE_CLICK, SETTINGS_SIGNAL
        try:
            delay_between_clicks = delay_between_clicks_entry.get()
            delay_before_click = delay_before_click_entry.get()

            if delay_between_clicks == "" or delay_before_click == "":
                raise ValueError("All fields must be filled out.")

            delay_between_clicks = float(delay_between_clicks)
            delay_before_click = float(delay_before_click)

            DELAY_BETWEEN_CLICKS = delay_between_clicks
            DELAY_BEFORE_CLICK = delay_before_click

            console.log(Text("Settings saved!", style="green"), highlight=True)
        except ValueError as e:
            console.log(Text(f"Invalid input: {e}", style="red"))
        else:
            settings_window.destroy()
            SETTINGS_SIGNAL = False

    def close_settings():
        global SETTINGS_SIGNAL
        SETTINGS_SIGNAL = False
        settings_window.destroy()

    settings_window = tk.Tk()
    settings_window.title("Settings")
    settings_window.configure(bg="#2e2e2e")
    settings_window.geometry("400x170")
    settings_window.attributes("-topmost", True)

    label_style = {"bg": "#2e2e2e", "fg": "#f0f0f0", "font": ("Helvetica", 10)}
    entry_style = {"bg": "#3e3e3e", "fg": "#f0f0f0", "insertbackground": "#f0f0f0", "font": ("Helvetica", 10)}
    button_style = {"bg": "#5e5e5e", "fg": "#f0f0f0", "font": ("Helvetica", 10)}

    tk.Label(settings_window, text="Delay Between Clicks (s)", **label_style).grid(row=0, column=0, padx=10, pady=5)
    delay_between_clicks_entry = tk.Entry(settings_window, **entry_style)
    delay_between_clicks_entry.grid(row=0, column=1, padx=10, pady=5)
    delay_between_clicks_entry.insert(0, str(DELAY_BETWEEN_CLICKS))

    tk.Label(settings_window, text="Delay Before Click (s)", **label_style).grid(row=1, column=0, padx=10, pady=5)
    delay_before_click_entry = tk.Entry(settings_window, **entry_style)
    delay_before_click_entry.grid(row=1, column=1, padx=10, pady=5)
    delay_before_click_entry.insert(0, str(DELAY_BEFORE_CLICK))

    tk.Button(settings_window, text="Save", command=save_settings, **button_style).grid(row=3, column=0, padx=10, pady=20)
    tk.Button(settings_window, text="Close", command=close_settings, **button_style).grid(row=3, column=1, padx=10, pady=20)

    settings_window.mainloop()

def main():
    global STOP_SIGNAL, SETTINGS_SIGNAL
    last_click_info = None

    messages_panel = MessagesPanel("", title="Messages", border_style="blue")
    console = CustomConsole(messages_panel=messages_panel)
    console.log(Text("Starting...", style="blue"), highlight=True)
    model, device = load_model(console)
    window = find_telegram_window(console)
    if not window:
        console.log("[red]Telegram window not found. Exiting...[/red]")
        return

    def keyboard_listener():
        while not STOP_SIGNAL:
            check_keyboard(console)
            time.sleep(0.1)

    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.start()

    bring_window_to_foreground(console, window)
    capture_gen = capture_telegram_window(console, window)
    frame_count = 0

    with Live(console=console, refresh_per_second=10) as live:
        while not STOP_SIGNAL:
            if SETTINGS_SIGNAL:
                show_settings(console)
                SETTINGS_SIGNAL = False
                continue

            start_time = time.time()
            screenshot, bbox = next(capture_gen)

            if frame_count % FRAME_SKIP == 0:
                preprocessed_frame = preprocess_image(screenshot, device)

                with torch.no_grad():
                    prediction = model(preprocessed_frame, augment=False, visualize=False)[0]

                predictions = non_max_suppression(prediction, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD, agnostic=False)
            else:
                predictions = []

            if predictions and predictions[0] is not None:
                for det in predictions[0]:
                    xyxy = det[:4].tolist()
                    score = det[4].item()
                    class_id = int(det[5].item())

                    if score > CONFIDENCE_THRESHOLD and class_id == TARGET_ID:
                        x1, y1, x2, y2 = map(int, xyxy)
                        width_scale = bbox["width"] / 416
                        height_scale = bbox["height"] / 416
                        x1, y1, x2, y2 = int(x1 * width_scale), int(y1 * height_scale), int(x2 * width_scale), int(y2 * height_scale)

                        x1, y1, x2, y2 = x1 + bbox["left"], y1 + bbox["top"], x2 + bbox["left"], y2 + bbox["top"]

                        click_key = (x1, y1, x2, y2)
                        if click_key not in click_counters:
                            click_counters[click_key] = 0

                        time.sleep(DELAY_BEFORE_CLICK)
                        last_click_info = perform_click(console, (x1 + x2) // 2, (y1 + y2) // 2)
                        time.sleep(DELAY_BETWEEN_CLICKS)

            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time if elapsed_time > 0 else float("inf")

            system_info = get_system_info()
            if last_click_info:
                x, y, count = last_click_info
                click_info_panel = Panel(f"Last click: ({x}, {y})\nClicks: {count}", title="Click Info", border_style="yellow")
            else:
                click_info_panel = Panel("No clicks", title="Click Info", border_style="yellow")

            system_info_panel = Panel(
                f"FPS: {fps:.2f}\nCPU: {system_info['CPU Usage']}\nRAM: {system_info['Memory Usage']}\nRAM Total: {system_info['Total Memory']}\nGPU: {system_info['GPU']}\nGPU Usage: {system_info['GPU Usage']}",
                title="System Info",
                border_style="green",
            )

            hotkeys_panel = Panel(Text("CTRL+Q - Exit\nCTRL+X - Pause/Resume\nCTRL+W - Settings"), title="Hotkeys", border_style="magenta")

            layout = Layout()
            layout.split_row(Layout(name="left"), Layout(messages_panel, name="right"))
            layout["left"].split_column(Layout(click_info_panel), Layout(system_info_panel), Layout(hotkeys_panel))

            live.update(layout)
            frame_count += 1

    cv2.destroyAllWindows()
    keyboard_thread.join()

if __name__ == "__main__":
    main()