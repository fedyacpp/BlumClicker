import os
import time
import threading
import queue
import json
import logging
import tkinter as tk
import cv2
import numpy as np
import torch
import psutil
import pygetwindow as gw
import win32process
import win32gui
import win32api
import win32con
import keyboard
import mss
import easyocr
from tkinter import filedialog, ttk
from ultralytics import YOLO
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from rich.text import Text
from rich.prompt import Prompt


class TelegramBot:

    CONFIDENCE_THRESHOLD = 0.6
    WINDOW_TITLE = "Telegram"
    FRAME_SKIP = 1
    TARGET_ID = 1
    SETTINGS_FILE = "settings.json"

    def __init__(self):
        self.stop_signal = False
        self.ctrl_q_pressed_once = False
        self.pause_signal = False
        self.settings_signal = False
        self.fps_lock = 60
        self.show_debug_window = False
        self.delay_between_clicks = 0
        self.delay_before_click = 0
        self.auto_play = False
        self.model_path = ""
        self.click_all_bombs = False

        self.model_lock = threading.Lock()
        self.click_counters = {}

        self.last_click_info = None
        self.debug_image = None

        logging.basicConfig(level=logging.INFO)
        logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

        self.messages_panel = self.MessagesPanel("", title="Messages", border_style="blue")
        self.console = self.CustomConsole(messages_panel=self.messages_panel)

        self.load_settings()

        self.model = self.load_model()

        self.window = self.find_telegram_window()
        if not self.window:
            self.console.log("[red]Telegram window not found. Exiting...[/red]")
            exit()

    class MessagesPanel(Panel):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_lines = 15
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

    def preprocess_image(self, image):
        return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 640))

    def find_telegram_window(self):
        self.console.log(Text(f"Searching for window '{self.WINDOW_TITLE}'...", style="blue"), highlight=True)
        windows = gw.getWindowsWithTitle(self.WINDOW_TITLE)
        if not windows:
            self.console.log(Text(f"Window '{self.WINDOW_TITLE}' not found", style="red"), highlight=True)
            return None
        self.console.log(Text(f"Window '{windows[0].title}' found!", style="green"), highlight=True)
        return windows[0]

    def capture_telegram_window(self):
        hwnd = self.window._hWnd
        with mss.mss() as sct:
            while not self.stop_signal:
                if self.pause_signal or self.settings_signal:
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

    def perform_click(self, x, y):
        win32api.SetCursorPos((x, y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

        click_key = (x, y)
        self.click_counters[click_key] = self.click_counters.get(click_key, 0) + 1

        return x, y, self.click_counters[click_key]

    def check_keyboard(self):
        while True:
            if keyboard.is_pressed("ctrl+q"):
                if not self.ctrl_q_pressed_once:
                    self.console.log(Text("CTRL+Q pressed. Press again to exit.", style="yellow"), highlight=True)
                    self.ctrl_q_pressed_once = True
                else:
                    self.console.log(Text("Exiting...", style="green"), highlight=True)
                    self.stop_signal = True
                    break
            elif keyboard.is_pressed("ctrl+x"):
                self.pause_signal = not self.pause_signal
                status = "paused" if self.pause_signal else "resumed"
                self.console.log(Text(f"Script {status}.", style="magenta"), highlight=True)
                time.sleep(0.5)
            elif keyboard.is_pressed("ctrl+w"):
                if not self.settings_signal:
                    self.console.log(Text("Opening settings panel...", style="cyan"), highlight=True)
                    self.settings_signal = True
            time.sleep(0.1)

    def get_system_info(self):
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

    def bring_window_to_foreground(self):
        hwnd = self.window._hWnd
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

            self.console.log(Text(f"Window '{self.window.title}' brought to the foreground!", style="green"), highlight=True)
        except Exception as e:
            self.console.log(Text(f"Error: {e}", style="red"), highlight=True)

    def load_settings(self):
        if not os.path.exists(self.SETTINGS_FILE):
            self.console.log("Settings file does not exist. Using default settings.")
            return

        try:
            with open(self.SETTINGS_FILE, "r") as file:
                settings = json.load(file)
                self.delay_between_clicks = settings.get("DELAY_BETWEEN_CLICKS", self.delay_between_clicks)
                self.delay_before_click = settings.get("DELAY_BEFORE_CLICK", self.delay_before_click)
                self.fps_lock = settings.get("FPS_LOCK", self.fps_lock)
                self.auto_play = settings.get("AUTO_PLAY", self.auto_play)
                self.model_path = settings.get("MODEL_PATH", self.model_path)
                self.show_debug_window = settings.get("SHOW_DEBUG_WINDOW", self.show_debug_window)
                self.click_all_bombs = settings.get("CLICK_ALL_BOMBS", self.click_all_bombs)
                self.console.log("Settings loaded successfully.")
        except json.JSONDecodeError:
            self.console.log("Error: JSON file is empty or invalid. Using default settings.")
        except Exception as e:
            self.console.log(f"Unexpected error loading settings: {e}. Using default settings.")

    def save_settings(self):
        settings = {
            "DELAY_BETWEEN_CLICKS": self.delay_between_clicks,
            "DELAY_BEFORE_CLICK": self.delay_before_click,
            "FPS_LOCK": self.fps_lock,
            "AUTO_PLAY": self.auto_play,
            "MODEL_PATH": self.model_path,
            "SHOW_DEBUG_WINDOW": self.show_debug_window,
            "CLICK_ALL_BOMBS": self.click_all_bombs
        }
        with open(self.SETTINGS_FILE, "w") as file:
            json.dump(settings, file, indent=4)

    def show_settings_panel(self):

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
            try:
                self.delay_between_clicks = float(delay_between_clicks_entry.get())
                self.delay_before_click = float(delay_before_click_entry.get())
                self.fps_lock = int(fps_lock_entry.get())
                self.auto_play = auto_play_var.get()
                self.model_path = model_path_entry.get()
                self.show_debug_window = show_debug_window_var.get()
                self.click_all_bombs = click_all_bombs_var.get()
                self.save_settings()
                self.console.log(Text("Settings updated!", style="green"), highlight=True)
            except ValueError:
                self.console.log(Text("Error: Invalid input. Please enter valid numbers.", style="red"), highlight=True)

            settings_window.destroy()
            self.settings_signal = False

        def cancel_and_close():
            settings_window.destroy()
            self.settings_signal = False
            self.console.log(Text("Settings unchanged.", style="yellow"), highlight=True)

        settings_window = create_settings_window()
        setup_styles()
        main_frame = create_main_frame(settings_window)

        delay_between_clicks_entry = create_entry(main_frame, 0, "Delay Between Clicks (seconds):", self.delay_between_clicks)
        delay_before_click_entry = create_entry(main_frame, 1, "Delay Before Click (seconds):", self.delay_before_click)
        fps_lock_entry = create_entry(main_frame, 2, "FPS Lock:", self.fps_lock)
        auto_play_var = create_checkbox(main_frame, 3, "Auto Play:", self.auto_play)
        show_debug_window_var = create_checkbox(main_frame, 4, "Show Debug Window:", self.show_debug_window)
        click_all_bombs_var = create_checkbox(main_frame, 5, "Click All Bombs:", self.click_all_bombs)

        ttk.Label(main_frame, text="Model Path:").grid(row=6, column=0, sticky="w", pady=10)
        model_path_entry = ttk.Entry(main_frame)
        model_path_entry.insert(0, self.model_path)
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

    def update_debug_window(self):
        if self.show_debug_window and self.debug_image is not None:
            cv2.imshow("Debug Window", self.debug_image)
            hwnd = win32gui.FindWindow(None, "Debug Window")
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            cv2.waitKey(1)
        else:
            if cv2.getWindowProperty("Debug Window", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Debug Window")

    def load_model(self):
        self.console.log(Text("Loading model...", style="blue"))
        if not self.model_path:
            default_model_path = os.path.join(os.path.dirname(__file__), "best.pt")
            self.model_path = Prompt.ask(Text("Path to model weights file", style="bold magenta"),
                                         default=default_model_path)
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading...", total=100)
            model = YOLO(self.model_path)
            progress.update(task, advance=100)
        self.console.log(Text("Model loaded!", style="green"), highlight=True)
        return model

    def detect_play_button(self, screenshot, bbox, results_queue):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(screenshot)

        for (bbox_coords, text, prob) in result:
            if "Play" in text:
                (top_left, top_right, bottom_right, bottom_left) = bbox_coords
                top_left = (int(top_left[0]), int(top_left[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                x, y, w, h = top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
                self.console.log(Text(f"'Play' detected at ({x, y, w, h})", style="green"), highlight=True)
                results_queue.put((x + w // 2, y + h // 2))
                return
        results_queue.put(None)

    def check_overlap(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        if y3 < y2 and x3 < x2 and x4 > x1:
            return True
        return False

    def run(self):
        self.bring_window_to_foreground()
        capture_gen = self.capture_telegram_window()
        frame_count = 0
        last_frame_time = time.time()

        keyboard_thread = threading.Thread(target=self.check_keyboard)
        keyboard_thread.start()

        with Live(console=self.console, refresh_per_second=self.fps_lock) as live:
            results_queue = queue.Queue()
            ocr_thread = None

            while not self.stop_signal:
                if self.settings_signal:
                    self.show_settings_panel()
                    continue

                start_time = time.time()
                image, bbox = next(capture_gen)

                if frame_count % self.FRAME_SKIP == 0:
                    preprocessed_frame = self.preprocess_image(image)

                    with self.model_lock:
                        with torch.no_grad():
                            prediction = self.model(preprocessed_frame, augment=False, visualize=False)
                else:
                    prediction = []

                if self.auto_play and (ocr_thread is None or not ocr_thread.is_alive()):
                    ocr_thread = threading.Thread(target=self.detect_play_button, args=(image, bbox, results_queue))
                    ocr_thread.start()

                try:
                    play_button_coords = results_queue.get_nowait()
                    if play_button_coords:
                        x, y = play_button_coords
                        self.perform_click(x + bbox["left"], y + bbox["top"])
                        time.sleep(self.delay_between_clicks)
                except queue.Empty:
                    pass

                if prediction:
                    det = prediction[0]
                    if det.boxes is not None:
                        boxes = det.boxes.xyxy.cpu().numpy()
                        scores = det.boxes.conf.cpu().numpy()
                        class_ids = det.boxes.cls.cpu().numpy().astype(int)

                        self.debug_image = image.copy()

                        sorted_indices = np.argsort(boxes[:, 1])

                        for i in sorted_indices:
                            box = boxes[i]
                            score = scores[i]
                            class_id = class_ids[i]

                            if score > self.CONFIDENCE_THRESHOLD:
                                x1, y1, x2, y2 = map(int, box)
                                width_scale = bbox["width"] / 640
                                height_scale = bbox["height"] / 640
                                x1, y1, x2, y2 = int(x1 * width_scale), int(y1 * height_scale), int(x2 * width_scale), int(y2 * height_scale)

                                color = (0, 255, 0) if class_id == self.TARGET_ID else (255, 0, 0)
                                cv2.rectangle(self.debug_image, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(self.debug_image, f'{class_id}: {score:.2f}', (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                                if class_id == self.TARGET_ID or (self.click_all_bombs and class_id == 0):
                                    overlapped = any(self.check_overlap(box, other_box) for other_box in boxes if not np.array_equal(box, other_box))

                                    if not overlapped:
                                        x1 += bbox["left"]
                                        y1 += bbox["top"]
                                        x2 += bbox["left"]
                                        y2 += bbox["top"]
                                        time.sleep(self.delay_before_click)
                                        self.last_click_info = self.perform_click((x1 + x2) // 2, (y1 + y2) // 2)
                                        time.sleep(self.delay_between_clicks)

                        self.update_debug_window()

                current_time = time.time()
                elapsed_time = current_time - last_frame_time
                fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')
                last_frame_time = current_time

                target_frame_time = 1 / self.fps_lock
                if elapsed_time < target_frame_time:
                    time.sleep(target_frame_time - elapsed_time)

                system_info = self.get_system_info()
                if self.last_click_info:
                    x, y, count = self.last_click_info
                    click_info_panel = Panel(f"Last click: ({x}, {y})\nClicks: {count}", title="Click Info", border_style="yellow")
                else:
                    click_info_panel = Panel("No clicks", title="Click Info", border_style="yellow")

                system_info_panel = Panel(
                    f"FPS: {fps:.2f}\nCPU: {system_info['CPU Usage']}\nRAM: {system_info['Memory Usage']}\nRAM Total: {system_info['Total Memory']}\nGPU: {system_info['GPU']}\nGPU Usage: {system_info['GPU Usage']}",
                    title="System Info",
                    border_style="green",
                )

                hotkeys_panel = Panel(Text("CTRL+Q - Exit\nCTRL+X - Pause/Resume\nCTRL+W - Settings"), title="Hotkeys",
                                      border_style="magenta")

                layout = Layout()
                layout.split_row(Layout(name="left"), Layout(self.messages_panel, name="right"))
                layout["left"].split_column(Layout(click_info_panel), Layout(system_info_panel), Layout(hotkeys_panel))

                live.update(layout)
                frame_count += 1

        keyboard_thread.join()
        if cv2.getWindowProperty("Debug Window", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    bot = TelegramBot()
    bot.run()
