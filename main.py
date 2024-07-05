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
import json
import easyocr
import queue
import logging
from ultralytics import YOLO
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.text import Text
from rich.prompt import Prompt
from tkinter import filedialog
from tkinter import ttk

CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.5
WINDOW_TITLE = "Telegram"
FRAME_SKIP = 1
TARGET_ID = 1
SETTINGS_FILE = "settings.json"

STOP_SIGNAL = False
CTRL_Q_PRESSED_ONCE = False
PAUSE_SIGNAL = False
SETTINGS_SIGNAL = False
FPS_LOCK = 60
SHOW_DEBUG_WINDOW = False
DELAY_BETWEEN_CLICKS = 0
DELAY_BEFORE_CLICK = 0
AUTO_PLAY = False
MODEL_PATH = ""
CLICK_ALL_BOMBS = False 

model_lock = threading.Lock()
play_button_lock = threading.Lock()

click_counters = {}

logging.basicConfig(level=logging.INFO)
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

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

def preprocess_image(image):
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 640))

def find_telegram_window(console, title_keyword=WINDOW_TITLE):
    console.log(Text(f"Searching for window '{title_keyword}'...", style="blue"), highlight=True)
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        console.log(Text(f"Window '{title_keyword}' not found", style="red"), highlight=True)
        return None
    console.log(Text(f"Window '{windows[0].title}' found!", style="green"), highlight=True)
    return windows[0]

def capture_telegram_window(console, window):
    global STOP_SIGNAL, PAUSE_SIGNAL, SETTINGS_SIGNAL
    hwnd = window._hWnd
    with mss.mss() as sct:
        while not STOP_SIGNAL:
            if PAUSE_SIGNAL or SETTINGS_SIGNAL:
                time.sleep(0.1)
                continue

            window_rect = win32gui.GetWindowRect(hwnd)
            bbox = {
                "left": window_rect[0],
                "top": window_rect[1],
                "width": window_rect[2] - window_rect[0],
                "height": window_rect[3] - window_rect[1],
            }

            screenshot = sct.grab(bbox)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            yield img, bbox

def perform_click(console, x, y):
    global click_counters
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    click_key = (x, y)
    click_counters[click_key] = click_counters.get(click_key, 0) + 1

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
            PAUSE_SIGNAL = not PAUSE_SIGNAL
            status = "paused" if PAUSE_SIGNAL else "resumed"
            console.log(Text(f"Script {status}.", style="magenta"), highlight=True)
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
            gpu_usage = f"{torch.cuda.memory_allocated(0) / gpu.total_memory * 100:.2f}%"
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

def load_settings():
    global DELAY_BETWEEN_CLICKS, DELAY_BEFORE_CLICK, FPS_LOCK, AUTO_PLAY, MODEL_PATH, SHOW_DEBUG_WINDOW, CLICK_ALL_BOMBS
    if not os.path.exists(SETTINGS_FILE):
        print("Settings file does not exist. Using default settings.")
        return

    try:
        with open(SETTINGS_FILE, "r") as file:
            settings = json.load(file)
            DELAY_BETWEEN_CLICKS = settings.get("DELAY_BETWEEN_CLICKS", DELAY_BETWEEN_CLICKS)
            DELAY_BEFORE_CLICK = settings.get("DELAY_BEFORE_CLICK", DELAY_BEFORE_CLICK)
            FPS_LOCK = settings.get("FPS_LOCK", FPS_LOCK)
            AUTO_PLAY = settings.get("AUTO_PLAY", AUTO_PLAY)
            MODEL_PATH = settings.get("MODEL_PATH", MODEL_PATH)
            SHOW_DEBUG_WINDOW = settings.get("SHOW_DEBUG_WINDOW", SHOW_DEBUG_WINDOW)
            CLICK_ALL_BOMBS = settings.get("CLICK_ALL_BOMBS", CLICK_ALL_BOMBS)
            print("Settings loaded successfully.")
    except json.JSONDecodeError:
        print("Error: JSON file is empty or invalid. Using default settings.")
    except Exception as e:
        print(f"Unexpected error loading settings: {e}. Using default settings.")

def save_settings():
    settings = {
        "DELAY_BETWEEN_CLICKS": DELAY_BETWEEN_CLICKS,
        "DELAY_BEFORE_CLICK": DELAY_BEFORE_CLICK,
        "FPS_LOCK": FPS_LOCK,
        "AUTO_PLAY": AUTO_PLAY,
        "MODEL_PATH": MODEL_PATH,
        "SHOW_DEBUG_WINDOW": SHOW_DEBUG_WINDOW,
        "CLICK_ALL_BOMBS": CLICK_ALL_BOMBS
    }
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file, indent=4)

def show_settings_panel(console):
    global DELAY_BETWEEN_CLICKS, DELAY_BEFORE_CLICK, FPS_LOCK, AUTO_PLAY, MODEL_PATH, SHOW_DEBUG_WINDOW, CLICK_ALL_BOMBS, SETTINGS_SIGNAL

    def create_settings_window():
        settings_window = tk.Toplevel()
        settings_window.title("Settings")
        settings_window.geometry("600x450")
        settings_window.configure(bg='#2E3B4E')
        settings_window.attributes('-topmost', True)
        return settings_window

    def setup_styles():
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", foreground="white", background="#2E3B4E", font=('Arial', 10))
        style.configure("TEntry", fieldbackground="#4A5B70", foreground="white", font=('Arial', 10))
        style.configure("TButton", background="#4A5B70", foreground="white", font=('Arial', 10))
        style.map("TButton", background=[('active', '#5A6B80')])
        style.configure("TCheckbutton", foreground="white", background="#2E3B4E", font=('Arial', 10))
        style.map("TCheckbutton", background=[('active', '#3E4B5E')])

    def create_main_frame(settings_window):
        main_frame = ttk.Frame(settings_window, padding="20", style="TLabel")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)
        return main_frame

    def create_entry(main_frame, row, text, value):
        ttk.Label(main_frame, text=text).grid(row=row, column=0, sticky="w", pady=10)
        entry = ttk.Entry(main_frame)
        entry.insert(0, str(value))
        entry.grid(row=row, column=1, sticky="ew", pady=10, padx=(10, 0))
        return entry

    def create_checkbox(main_frame, row, text, value):
        ttk.Label(main_frame, text=text).grid(row=row, column=0, sticky="w", pady=10)
        var = tk.BooleanVar(value=value)
        ttk.Checkbutton(main_frame, variable=var).grid(row=row, column=1, sticky="w", pady=10, padx=(10, 0))
        return var

    def browse_file():
        filename = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if filename:
            model_path_entry.delete(0, tk.END)
            model_path_entry.insert(0, filename)

    def save_and_close():
        global DELAY_BETWEEN_CLICKS, DELAY_BEFORE_CLICK, FPS_LOCK, AUTO_PLAY, MODEL_PATH, SHOW_DEBUG_WINDOW, CLICK_ALL_BOMBS, SETTINGS_SIGNAL
        try:
            DELAY_BETWEEN_CLICKS = float(delay_between_clicks_entry.get())
            DELAY_BEFORE_CLICK = float(delay_before_click_entry.get())
            FPS_LOCK = int(fps_lock_entry.get())
            AUTO_PLAY = auto_play_var.get()
            MODEL_PATH = model_path_entry.get()
            SHOW_DEBUG_WINDOW = show_debug_window_var.get()
            CLICK_ALL_BOMBS = click_all_bombs_var.get()
            save_settings()
            console.log(Text("Settings updated!", style="green"), highlight=True)
        except ValueError:
            console.log(Text("Error: Invalid input. Please enter valid numbers.", style="red"), highlight=True)

        settings_window.destroy()
        SETTINGS_SIGNAL = False

    def cancel_and_close():
        global SETTINGS_SIGNAL
        settings_window.destroy()
        SETTINGS_SIGNAL = False
        console.log(Text("Settings unchanged.", style="yellow"), highlight=True)

    settings_window = create_settings_window()
    setup_styles()
    main_frame = create_main_frame(settings_window)

    delay_between_clicks_entry = create_entry(main_frame, 0, "Delay Between Clicks (seconds):", DELAY_BETWEEN_CLICKS)
    delay_before_click_entry = create_entry(main_frame, 1, "Delay Before Click (seconds):", DELAY_BEFORE_CLICK)
    fps_lock_entry = create_entry(main_frame, 2, "FPS Lock:", FPS_LOCK)
    auto_play_var = create_checkbox(main_frame, 3, "Auto Play:", AUTO_PLAY)
    show_debug_window_var = create_checkbox(main_frame, 4, "Show Debug Window:", SHOW_DEBUG_WINDOW)
    click_all_bombs_var = create_checkbox(main_frame, 5, "Click All Bombs:", CLICK_ALL_BOMBS)

    ttk.Label(main_frame, text="Model Path:").grid(row=6, column=0, sticky="w", pady=10)
    model_path_entry = ttk.Entry(main_frame)
    model_path_entry.insert(0, MODEL_PATH)
    model_path_entry.grid(row=6, column=1, sticky="ew", pady=10, padx=(10, 0))
    ttk.Button(main_frame, text="Browse", command=browse_file).grid(row=6, column=2, pady=10, padx=(10, 0))

    button_frame = ttk.Frame(main_frame, style="TLabel")
    button_frame.grid(row=7, column=0, columnspan=3, pady=20)

    ttk.Button(button_frame, text="Save", command=save_and_close).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="Cancel", command=cancel_and_close).pack(side=tk.LEFT)

    settings_window.protocol("WM_DELETE_WINDOW", cancel_and_close)
    settings_window.grab_set()
    settings_window.focus_set()
    settings_window.wait_window()

def update_debug_window(debug_image):
    if SHOW_DEBUG_WINDOW:
        cv2.imshow("Debug Window", debug_image)
        hwnd = win32gui.FindWindow(None, "Debug Window")
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        cv2.waitKey(1)
    else:
        if cv2.getWindowProperty("Debug Window", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Debug Window")

def load_model(console):
    global MODEL_PATH
    console.log(Text("Loading model...", style="blue"))
    if not MODEL_PATH:
        default_model_path = os.path.join(os.path.dirname(__file__), "best.pt")
        MODEL_PATH = Prompt.ask(Text("Path to model weights file", style="bold magenta"), default=default_model_path)
    with Progress() as progress:
        task = progress.add_task("[cyan]Loading...", total=100)
        model = YOLO(MODEL_PATH)
        progress.update(task, advance=100)
    console.log(Text("Model loaded!", style="green"), highlight=True)
    return model

def detect_play_button(console, screenshot, bbox, results_queue):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(screenshot)
    
    for (bbox, text, prob) in result:
        if "Play" in text:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            x, y, w, h = top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
            console.log(Text(f"'Play' detected at ({x, y, w, h})", style="green"), highlight=True)
            results_queue.put((x + w // 2, y + h // 2))
            return
    results_queue.put(None)

def update_debug_window(debug_image):
    if SHOW_DEBUG_WINDOW:
        cv2.imshow("Debug Window", debug_image)
        hwnd = win32gui.FindWindow(None, "Debug Window")
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        cv2.waitKey(1)
    else:
        if cv2.getWindowProperty("Debug Window", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Debug Window")

def main():
    global STOP_SIGNAL, SETTINGS_SIGNAL, FPS_LOCK, AUTO_PLAY, SHOW_DEBUG_WINDOW, CLICK_ALL_BOMBS
    last_click_info = None

    load_settings()

    messages_panel = MessagesPanel("", title="Messages", border_style="blue")
    console = CustomConsole(messages_panel=messages_panel)
    console.log(Text("Starting...", style="blue"), highlight=True)
    model = load_model(console)
    window = find_telegram_window(console)
    if not window:
        console.log("[red]Telegram window not found. Exiting...[/red]")
        return

    keyboard_thread = threading.Thread(target=check_keyboard, args=(console,))
    keyboard_thread.start()

    bring_window_to_foreground(console, window)
    capture_gen = capture_telegram_window(console, window)
    frame_count = 0
    last_frame_time = time.time()

    with Live(console=console, refresh_per_second=FPS_LOCK) as live:
        results_queue = queue.Queue()
        ocr_thread = None
        
        while not STOP_SIGNAL:
            if SETTINGS_SIGNAL:
                show_settings_panel(console)
                continue

            start_time = time.time()
            image, bbox = next(capture_gen)

            if frame_count % FRAME_SKIP == 0:
                preprocessed_frame = preprocess_image(image)

                with model_lock:
                    with torch.no_grad():
                        prediction = model(preprocessed_frame, augment=False, visualize=False)
            else:
                prediction = []

            if AUTO_PLAY and (ocr_thread is None or not ocr_thread.is_alive()):
                ocr_thread = threading.Thread(target=detect_play_button, args=(console, image, bbox, results_queue))
                ocr_thread.start()

            try:
                play_button_coords = results_queue.get_nowait()
                if play_button_coords:
                    x, y = play_button_coords
                    perform_click(console, x + bbox["left"], y + bbox["top"])
                    time.sleep(DELAY_BETWEEN_CLICKS)
            except queue.Empty:
                pass

            if prediction:
                det = prediction[0]
                if det.boxes is not None:
                    boxes = det.boxes.xyxy.cpu().numpy()
                    scores = det.boxes.conf.cpu().numpy()
                    class_ids = det.boxes.cls.cpu().numpy().astype(int)

                    debug_image = image.copy()

                    for i, box in enumerate(boxes):
                        score = scores[i]
                        class_id = class_ids[i]

                        if score > CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box)
                            width_scale = bbox["width"] / 640
                            height_scale = bbox["height"] / 640
                            x1, y1, x2, y2 = int(x1 * width_scale), int(y1 * height_scale), int(x2 * width_scale), int(y2 * height_scale)

                            color = (0, 255, 0) if class_id == TARGET_ID else (255, 0, 0)
                            cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(debug_image, f'{class_id}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                            if class_id == TARGET_ID or (CLICK_ALL_BOMBS and class_id == 0):
                                x1, y1, x2, y2 = x1 + bbox["left"], y1 + bbox["top"], x2 + bbox["left"], y2 + bbox["top"]
                                time.sleep(DELAY_BEFORE_CLICK)
                                last_click_info = perform_click(console, (x1 + x2) // 2, (y1 + y2) // 2)
                                time.sleep(DELAY_BETWEEN_CLICKS)

                    update_debug_window(debug_image)

            current_time = time.time()
            elapsed_time = current_time - last_frame_time
            fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')
            last_frame_time = current_time

            target_frame_time = 1 / FPS_LOCK
            if elapsed_time < target_frame_time:
                time.sleep(target_frame_time - elapsed_time)

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
            if SHOW_DEBUG_WINDOW:
                cv2.imshow("Debug Window", debug_image)
                cv2.waitKey(1)
            else:
                if cv2.getWindowProperty("Debug Window", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Debug Window")

            hotkeys_panel = Panel(Text("CTRL+Q - Exit\nCTRL+X - Pause/Resume\nCTRL+W - Settings"), title="Hotkeys", border_style="magenta")

            layout = Layout()
            layout.split_row(Layout(name="left"), Layout(messages_panel, name="right"))
            layout["left"].split_column(Layout(click_info_panel), Layout(system_info_panel), Layout(hotkeys_panel))

            live.update(layout)
            frame_count += 1

    keyboard_thread.join()
    if cv2.getWindowProperty("Debug Window", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()