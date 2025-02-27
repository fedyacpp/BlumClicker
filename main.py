import os
import time
import threading
import queue
import json
import logging
import cv2
import numpy as np
import torch
import pygetwindow as gw
import win32process
import win32gui
import win32api
import win32con
import keyboard
import mss
import easyocr
import traceback
import sys
import psutil
from typing import Optional, Tuple, Dict, Generator

from PyQt5 import QtWidgets, QtCore
from ultralytics import YOLO
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.prompt import Prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("bot_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)


def check_dependencies():
    required_packages = {
        "opencv-python": "cv2",
        "numpy": "numpy",
        "torch": "torch",
        "pygetwindow": "pygetwindow",
        "pywin32": "win32gui",
        "keyboard": "keyboard",
        "mss": "mss",
        "easyocr": "easyocr",
        "PyQt5": "PyQt5",
        "ultralytics": "ultralytics",
        "rich": "rich"
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Отсутствуют необходимые пакеты: {', '.join(missing_packages)}")
        print("Установите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


class RichPanelLoggingHandler(logging.Handler):
    def __init__(self, custom_console: Console):
        super().__init__()
        self.console = custom_console

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.console.log(msg)
        except Exception:
            self.handleError(record)


class BlumClicker:
    CONFIDENCE_THRESHOLD = 0.7
    WINDOW_TITLES = ["Blum", "Telegram"]
    FRAME_SKIP = 1
    TARGET_ID = 1
    SETTINGS_FILE = "settings.json"
    LOG_FILE = "bot_log.txt"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        self.stop_signal = False
        self.pause_signal = False
        self.settings_signal = False
        self.is_running = False

        self.fps_lock = 60
        self.show_debug_window = False
        self.delay_between_clicks = 0.0
        self.delay_before_click = 0.0
        self.auto_play = True
        self.model_path = ""
        self.click_all_bombs = False
        self.retry_count = 3
        self.use_cpu = torch.cuda.is_available() == False
        self.enable_sound = False

        self.model_lock = threading.Lock()
        self.ui_lock = threading.Lock()
        self.click_lock = threading.Lock()
        
        self.click_counters = {}
        self.last_click_info = None
        self.debug_image = None
        self.error_count = {}
        self.last_errors = []
        self.start_time = time.time()

        self.messages_panel = self.MessagesPanel("", title="Сообщения", border_style="blue")
        self.console = self.CustomConsole(messages_panel=self.messages_panel, force_terminal=True, color_system="auto")
        
        handler = RichPanelLoggingHandler(self.console)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self.load_settings()
        
        try:
            self.model = self.load_model()
        except Exception as e:
            self.console.log(f"[red]Критическая ошибка при загрузке модели: {e}[/red]")
            logging.error(f"Критическая ошибка при загрузке модели: {e}")
            print(f"ОШИБКА: Не удалось загрузить модель. {e}")
            sys.exit(1)

        try:
            self.console.log("[blue]Инициализация OCR...[/blue]")
            use_gpu = torch.cuda.is_available() and not self.use_cpu
            self.console.log(f"Использование GPU для OCR: {'Да' if use_gpu else 'Нет'}")
            self.ocr_reader = easyocr.Reader(["en", "ru"], gpu=use_gpu)
            self.console.log("[green]OCR инициализирован успешно![/green]")
        except Exception as e:
            self.console.log(f"[red]Ошибка при инициализации OCR: {e}[/red]")
            logging.error(f"Ошибка инициализации OCR: {e}")
            print(f"ОШИБКА: Не удалось инициализировать OCR. {e}")
            self.ocr_reader = None

        self.window = self.find_telegram_window()
        if not self.window:
            self.console.log("[red]Окно Telegram не найдено. Убедитесь, что приложение запущено и окно видимо.[/red]")
            logging.error("Окно Telegram не найдено. Программа будет закрыта.")
            print("ОШИБКА: Окно Telegram не найдено. Проверьте, что приложение запущено.")
            sys.exit(1)

        self.keyboard_thread = threading.Thread(target=self.check_keyboard, daemon=True)
        self.keyboard_thread.start()

    class MessagesPanel(Panel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_lines = 15
            self.messages = []

        def add_message(self, message: str) -> None:
            self.messages.append(message)
            if len(self.messages) > self.max_lines * 2:
                self.messages = self.messages[-self.max_lines:]
            self.renderable = Text.from_markup("\n".join(self.messages[-self.max_lines:]))

    class CustomConsole(Console):
        def __init__(self, *args, messages_panel=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.messages_panel = messages_panel

        def log(self, message: str, **kwargs) -> None:
            if self.messages_panel:
                timestamp = time.strftime('%H:%M:%S')
                formatted_message = f"[{timestamp}] {message}"
                self.messages_panel.add_message(formatted_message)
            super().log(message, markup=True, **kwargs)

    def log(self, *objects, highlight: bool = False, **kwargs) -> None:
        msg_str = " ".join(str(obj) for obj in objects)
        message = Text.from_markup(msg_str)
        if highlight:
            message.stylize("bold")
        if self.messages_panel:
            formatted_message = f"[{time.strftime('%H:%M:%S')}] {message.markup}"
            self.messages_panel.add_message(formatted_message)
        else:
            super().log(message, **kwargs)

    def load_settings(self) -> None:
        if not os.path.exists(self.SETTINGS_FILE):
            self.console.log("[yellow]Файл настроек не существует. Используются настройки по умолчанию.[/yellow]")
            logging.warning(f"'{self.SETTINGS_FILE}' не найден; используются значения по умолчанию.")
            self.save_settings()
            return
        
        try:
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as file:
                settings = json.load(file)
                
                self.delay_between_clicks = max(0, float(settings.get("DELAY_BETWEEN_CLICKS", self.delay_between_clicks)))
                self.delay_before_click = max(0, float(settings.get("DELAY_BEFORE_CLICK", self.delay_before_click)))
                self.fps_lock = max(1, int(settings.get("FPS_LOCK", self.fps_lock)))
                self.auto_play = bool(settings.get("AUTO_PLAY", self.auto_play))
                self.use_cpu = bool(settings.get("USE_CPU", self.use_cpu))
                self.enable_sound = bool(settings.get("ENABLE_SOUND", self.enable_sound))
                
                model_path = settings.get("MODEL_PATH", self.model_path)
                if model_path and not os.path.exists(model_path):
                    self.console.log(f"[yellow]Предупреждение: Указанный путь к модели не существует: {model_path}[/yellow]")
                    logging.warning(f"Путь к модели не существует: {model_path}")
                else:
                    self.model_path = model_path
                
                self.show_debug_window = bool(settings.get("SHOW_DEBUG_WINDOW", self.show_debug_window))
                self.click_all_bombs = bool(settings.get("CLICK_ALL_BOMBS", self.click_all_bombs))
                self.retry_count = int(settings.get("RETRY_COUNT", self.retry_count))
                
                self.console.log("[green]Настройки успешно загружены.[/green]")
                logging.info("Настройки загружены из файла.")
        except json.JSONDecodeError as e:
            self.console.log(f"[red]Ошибка: Неверный формат JSON файла. Строка: {e.lineno}, Позиция: {e.colno}[/red]")
            self.console.log("[yellow]Используются настройки по умолчанию.[/yellow]")
            logging.warning(f"Неверный JSON в '{self.SETTINGS_FILE}': {e}. Используются значения по умолчанию.")
        except (ValueError, TypeError) as e:
            self.console.log(f"[red]Ошибка при конвертации значений в настройках: {e}[/red]")
            self.console.log("[yellow]Используются настройки по умолчанию.[/yellow]")
            logging.warning(f"Ошибка валидации в настройках: {e}. Используются значения по умолчанию.")
        except Exception as e:
            self.console.log(f"[red]Непредвиденная ошибка при загрузке настроек: {e}[/red]")
            self.console.log("[yellow]Используются настройки по умолчанию.[/yellow]")
            logging.warning(f"Непредвиденная ошибка при загрузке настроек: {e}. Используются значения по умолчанию.")

    def save_settings(self) -> None:
        settings = {
            "DELAY_BETWEEN_CLICKS": self.delay_between_clicks,
            "DELAY_BEFORE_CLICK": self.delay_before_click,
            "FPS_LOCK": self.fps_lock,
            "AUTO_PLAY": self.auto_play,
            "MODEL_PATH": self.model_path,
            "SHOW_DEBUG_WINDOW": self.show_debug_window,
            "CLICK_ALL_BOMBS": self.click_all_bombs,
            "RETRY_COUNT": self.retry_count,
            "USE_CPU": self.use_cpu,
            "ENABLE_SOUND": self.enable_sound,
        }
        try:
            if os.path.exists(self.SETTINGS_FILE):
                backup_file = f"{self.SETTINGS_FILE}.bak"
                try:
                    with open(self.SETTINGS_FILE, "r", encoding="utf-8") as src:
                        with open(backup_file, "w", encoding="utf-8") as dst:
                            dst.write(src.read())
                except Exception as e:
                    self.console.log(f"[yellow]Не удалось создать резервную копию настроек: {e}[/yellow]")
            
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as file:
                json.dump(settings, file, indent=4, ensure_ascii=False)
            self.console.log("[green]Настройки успешно сохранены.[/green]")
            logging.info("Настройки сохранены в файл.")
        except PermissionError:
            self.console.log(f"[red]Ошибка доступа: Не удалось сохранить настройки. Проверьте права доступа к файлу {self.SETTINGS_FILE}[/red]")
            logging.error(f"Ошибка прав доступа при сохранении настроек в {self.SETTINGS_FILE}")
        except Exception as e:
            self.console.log(f"[red]Ошибка при сохранении настроек: {e}[/red]")
            logging.error(f"Ошибка при сохранении настроек: {e}")

    def load_model(self) -> YOLO:
        self.console.log("[blue]Загрузка модели...[/blue]")
        
        if not self.model_path:
            default_model_path = os.path.join(os.getcwd(), "best.pt")
            self.model_path = Prompt.ask("Путь к файлу модели", default=default_model_path)
        
        if not os.path.exists(self.model_path):
            self.console.log(f"[red]Файл модели не найден по пути: {self.model_path}[/red]")
            logging.error(f"Файл модели не найден по пути: {self.model_path}")
            
            model_files = [f for f in os.listdir() if f.endswith('.pt')]
            if model_files:
                self.console.log("[yellow]Найдены следующие файлы моделей в текущей директории:[/yellow]")
                for i, file in enumerate(model_files):
                    self.console.log(f"  {i+1}. {file}")
                
                try:
                    choice = Prompt.ask("Выберите номер файла или введите полный путь к модели", default="1")
                    if choice.isdigit() and 1 <= int(choice) <= len(model_files):
                        self.model_path = model_files[int(choice) - 1]
                    else:
                        self.model_path = choice
                except Exception as e:
                    self.console.log(f"[red]Ошибка при выборе модели: {e}[/red]")
                    raise ValueError(f"Не удалось выбрать модель: {e}")
            else:
                self.console.log("[red]Модели .pt не найдены в текущей директории.[/red]")
                raise FileNotFoundError(f"Модель не найдена по пути: {self.model_path} и в текущей директории")
        
        try:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Загрузка модели..."),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            )
            
            with progress:
                task = progress.add_task("[cyan]Загрузка модели...", total=100)
                
                def update_progress():
                    for i in range(1, 95):
                        progress.update(task, completed=i)
                        time.sleep(0.05)
                
                progress_thread = threading.Thread(target=update_progress)
                progress_thread.daemon = True
                progress_thread.start()
                
                with self.model_lock:
                    if self.use_cpu:
                        model = YOLO(self.model_path, task='detect', device='cpu')
                    else:
                        model = YOLO(self.model_path, task='detect')
                
                progress.update(task, completed=100)
                time.sleep(0.5)
            
            model_info = f"Модель {os.path.basename(self.model_path)}"
            try:
                class_names = model.names
                class_count = len(class_names)
                model_info += f" загружена успешно! Обнаружено {class_count} классов."
                if class_count > 0:
                    model_info += f" Классы: {', '.join([f'{i}: {name}' for i, name in class_names.items()])}"
            except Exception:
                model_info += " загружена успешно!"
            
            self.console.log(f"[green]{model_info}[/green]")
            logging.info(f"Модель загружена из {self.model_path}")
            return model
            
        except Exception as e:
            error_msg = f"Не удалось загрузить модель из {self.model_path}: {e}"
            self.console.log(f"[red]{error_msg}[/red]")
            logging.error(error_msg)
            
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                self.console.log("[yellow]Проблема с GPU. Попробуем использовать CPU...[/yellow]")
                try:
                    self.use_cpu = True
                    model = YOLO(self.model_path, task='detect', device='cpu')
                    self.console.log("[green]Модель успешно загружена на CPU![/green]")
                    return model
                except Exception as cpu_error:
                    self.console.log(f"[red]Не удалось загрузить модель на CPU: {cpu_error}[/red]")
            
            raise RuntimeError(error_msg)

    def find_telegram_window(self) -> Optional[gw.Win32Window]:
        self.console.log(f"[blue]Поиск окна Telegram...[/blue]")
        
        for title in self.WINDOW_TITLES:
            windows = gw.getWindowsWithTitle(title)
            visible_windows = [w for w in windows if w.visible]
            
            if visible_windows:
                window = visible_windows[0]
                self.console.log(f"[green]Найдено окно '{window.title}'![/green]")
                logging.info(f"Найдено окно: {window.title}")
                return window
            elif windows:
                window = windows[0]
                self.console.log(f"[yellow]Найдено окно '{window.title}', но оно может быть не видимым.[/yellow]")
                logging.info(f"Найдено окно (не видимое): {window.title}")
                return window
        
        self.console.log("[yellow]Окна с известными заголовками не найдены. Поиск любого окна Telegram...[/yellow]")
        all_windows = gw.getAllWindows()
        for window in all_windows:
            if 'telegram' in window.title.lower() in window.title.lower():
                self.console.log(f"[green]Найдено окно '{window.title}'![/green]")
                return window
                
        return None

    def bring_window_to_foreground(self) -> bool:
        if not self.window:
            self.console.log("[red]Окно не найдено. Невозможно вывести на передний план.[/red]")
            return False
        
        hwnd = self.window._hWnd
        for attempt in range(self.retry_count):
            try:
                window_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
                current_thread = win32api.GetCurrentThreadId()
                
                attached = False
                if window_thread != current_thread:
                    attached = win32process.AttachThreadInput(window_thread, current_thread, True)
                
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                
                win32gui.SetForegroundWindow(hwnd)
                win32gui.SetFocus(hwnd)
                
                if attached:
                    win32process.AttachThreadInput(window_thread, current_thread, False)
                
                foreground_hwnd = win32gui.GetForegroundWindow()
                if foreground_hwnd == hwnd:
                    self.console.log(f"[green]Окно '{self.window.title}' выведено на передний план![/green]")
                    logging.info("Окно Telegram выведено на передний план.")
                    return True
                else:
                    self.console.log(f"[yellow]Попытка {attempt+1}/{self.retry_count}: Не удалось вывести окно на передний план.[/yellow]")
                    time.sleep(0.5)
            except Exception as e:
                self.console.log(f"[red]Ошибка при выведении окна на передний план: {e}[/red]")
                logging.warning(f"Ошибка при выведении окна на передний план: {e}")
                time.sleep(0.5)
        
        self.console.log("[red]Не удалось вывести окно на передний план после нескольких попыток.[/red]")
        return False

    def capture_telegram_window(self) -> Generator[Tuple[np.ndarray, Dict[str, int]], None, None]:
        if not self.window:
            self.console.log("[red]Окно не найдено. Невозможно захватить изображение.[/red]")
            return
        
        hwnd = self.window._hWnd
        last_error_time = 0
        consecutive_errors = 0
        
        with mss.mss() as sct:
            while not self.stop_signal:
                if self.pause_signal or self.settings_signal:
                    time.sleep(0.1)
                    continue
                
                try:
                    window_rect = win32gui.GetWindowRect(hwnd)
                    
                    if window_rect[0] < -32000 or window_rect[1] < -32000:
                        if time.time() - last_error_time > 5:
                            self.console.log("[yellow]Окно минимизировано или скрыто. Ожидание...[/yellow]")
                            last_error_time = time.time()
                        time.sleep(0.5)
                        continue
                    
                    width = window_rect[2] - window_rect[0]
                    height = window_rect[3] - window_rect[1]
                    
                    if width <= 0 or height <= 0:
                        if time.time() - last_error_time > 5:
                            self.console.log(f"[yellow]Некорректные размеры окна: {width}x{height}. Ожидание...[/yellow]")
                            last_error_time = time.time()
                        time.sleep(0.5)
                        continue
                    
                    bbox = {
                        "left": window_rect[0],
                        "top": window_rect[1],
                        "width": width,
                        "height": height,
                    }
                    
                    screenshot = sct.grab(bbox)
                    img = np.array(screenshot)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    consecutive_errors = 0
                    
                    yield img, bbox
                    
                except Exception as e:
                    consecutive_errors += 1
                    now = time.time()
                    if now - last_error_time > 2:
                        self.console.log(f"[red]Ошибка при захвате экрана: {e}[/red]")
                        logging.error(f"Ошибка при захвате экрана: {e}")
                        last_error_time = now
                    
                    if consecutive_errors > 5:
                        self.console.log("[yellow]Слишком много ошибок подряд. Переподключение к окну...[/yellow]")
                        new_window = self.find_telegram_window()
                        if new_window:
                            self.window = new_window
                            hwnd = self.window._hWnd
                            self.console.log("[green]Окно найдено заново![/green]")
                        consecutive_errors = 0
                    
                    time.sleep(0.5)

    def perform_click(self, x: int, y: int) -> Tuple[int, int, int]:
        try:
            with self.click_lock:
                screen_width = win32api.GetSystemMetrics(0)
                screen_height = win32api.GetSystemMetrics(1)
                
                x = max(0, min(int(x), screen_width - 1))
                y = max(0, min(int(y), screen_height - 1))
                
                win32api.SetCursorPos((x, y))
                
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.01)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                
                click_key = (x, y)
                self.click_counters[click_key] = self.click_counters.get(click_key, 0) + 1
                
                if self.enable_sound:
                    win32api.MessageBeep(win32con.MB_OK)
                
                return x, y, self.click_counters[click_key]
                
        except Exception as e:
            self.console.log(f"[red]Ошибка при выполнении клика: {e}[/red]")
            logging.warning(f"Не удалось выполнить клик по координатам ({x},{y}): {e}")
            error_type = type(e).__name__
            self.error_count[error_type] = self.error_count.get(error_type, 0) + 1
            self.last_errors.append(f"{error_type}: {str(e)}")
            if len(self.last_errors) > 5:
                self.last_errors.pop(0)
            return x, y, -1

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, (640, 640))
        except Exception as e:
            self.console.log(f"[red]Ошибка при предобработке изображения: {e}[/red]")
            logging.error(f"Ошибка предобработки изображения: {e}")
            return np.zeros((640, 640, 3), dtype=np.uint8)

    def update_debug_window(self) -> None:
        if self.show_debug_window and self.debug_image is not None:
            try:
                h, w = self.debug_image.shape[:2]
                max_size = 1200
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_size = (int(w * scale), int(h * scale))
                    resized_img = cv2.resize(self.debug_image, new_size)
                else:
                    resized_img = self.debug_image
                
                cv2.imshow("Debug Window", resized_img)
                
                hwnd = win32gui.FindWindow(None, "Debug Window")
                if hwnd != 0:
                    win32gui.SetWindowPos(
                        hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
                    )
                cv2.waitKey(1)
            except Exception as e:
                self.console.log(f"[yellow]Ошибка при обновлении окна отладки: {e}[/yellow]")
                logging.warning(f"Ошибка при обновлении окна отладки: {e}")
        else:
            try:
                if cv2.getWindowProperty("Debug Window", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Debug Window")
            except:
                pass

    def detect_play_button(self, screenshot: np.ndarray, bbox: dict, results_queue: queue.Queue) -> None:
        if self.ocr_reader is None:
            results_queue.put(None)
            return
            
        try:
            debug_img = screenshot.copy() if self.show_debug_window else None
            
            result = self.ocr_reader.readtext(screenshot)
            
            found = False
            keywords = ["Play", "Играть", "PLAY", "ИГРАТЬ"]
            
            for (bbox_coords, text, prob) in result:
                text_lower = text.lower()
                
                if any(keyword.lower() in text_lower for keyword in keywords) and prob > 0.5:
                    (top_left, _, bottom_right, _) = bbox_coords
                    x, y = int(top_left[0]), int(top_left[1])
                    w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            debug_img, f"{text} ({prob:.2f})", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                    
                    self.console.log(f"[green]Кнопка '{text}' обнаружена с уверенностью {prob:.2f} по координатам ({x}, {y}, {w}, {h})[/green]")
                    results_queue.put((x + w // 2, y + h // 2))
                    found = True
                    break
            
            if debug_img is not None and self.show_debug_window:
                cv2.imshow("OCR Debug", debug_img)
                cv2.waitKey(1)
            
            if not found:
                results_queue.put(None)
                
        except Exception as e:
            self.console.log(f"[red]Ошибка OCR: {e}[/red]")
            logging.warning(f"Ошибка OCR: {e}")
            logging.debug(f"Trace: {traceback.format_exc()}")
            results_queue.put(None)

    def get_system_info(self) -> Dict[str, str]:
        try:
            cpu_usage = psutil.cpu_percent()
            
            mem = psutil.virtual_memory()
            mem_usage = mem.percent
            mem_total = mem.total / (1024 ** 3)
            mem_used = mem.used / (1024 ** 3)
            
            try:
                if torch.cuda.is_available() and not self.use_cpu:
                    gpu = torch.cuda.get_device_properties(0)
                    gpu_info = f"{gpu.name} - {gpu.total_memory // (1024 ** 2)} MB"
                    gpu_usage = f"{torch.cuda.memory_allocated(0) / gpu.total_memory * 100:.2f}%"
                    gpu_temp = "N/A"
                else:
                    gpu_info = "CPU режим"
                    gpu_usage = "N/A"
                    gpu_temp = "N/A"
            except Exception as e:
                gpu_info = f"Ошибка GPU: {type(e).__name__}"
                gpu_usage = "N/A"
                gpu_temp = "N/A"
                logging.warning(f"Ошибка при получении информации о GPU: {e}")
            
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / (1024 ** 2)
            process_cpu = process.cpu_percent()
            
            uptime = time.time() - self.start_time
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{int(hours)}ч {int(minutes)}м {int(seconds)}с"
            
            clicks_total = sum(self.click_counters.values())
            clicks_per_minute = clicks_total / (uptime / 60) if uptime > 0 else 0
            
            return {
                "CPU": f"{cpu_usage}%",
                "RAM": f"{mem_usage}% ({mem_used:.1f}/{mem_total:.1f} ГБ)",
                "Процесс": f"{process_memory:.1f} МБ, CPU: {process_cpu}%",
                "GPU": gpu_info,
                "GPU загрузка": gpu_usage,
                "Время работы": uptime_str,
                "Всего кликов": f"{clicks_total}",
                "Скорость": f"{clicks_per_minute:.1f} кликов/мин",
                "Ошибок": f"{sum(self.error_count.values())}"
            }
        except Exception as e:
            logging.error(f"Ошибка при получении системной информации: {e}")
            return {"Ошибка": f"Не удалось получить системную информацию: {e}"}

    def check_keyboard(self) -> None:
        try:
            keyboard.add_hotkey('ctrl+q', lambda: self._handle_hotkey('exit'))
            keyboard.add_hotkey('ctrl+x', lambda: self._handle_hotkey('pause'))
            keyboard.add_hotkey('ctrl+w', lambda: self._handle_hotkey('settings'))
            keyboard.add_hotkey('ctrl+d', lambda: self._handle_hotkey('debug'))
            keyboard.add_hotkey('ctrl+r', lambda: self._handle_hotkey('reload'))
            keyboard.add_hotkey('ctrl+f', lambda: self._handle_hotkey('find_window'))
            keyboard.add_hotkey('ctrl+a', lambda: self._handle_hotkey('toggle_autoplay'))
            keyboard.add_hotkey('ctrl+s', lambda: self._handle_hotkey('toggle_sound'))
            
            self.console.log("[green]Горячие клавиши зарегистрированы успешно.[/green]")
            
            while not self.stop_signal:
                time.sleep(0.1)
                
        except Exception as e:
            self.console.log(f"[red]Ошибка в мониторинге клавиатуры: {e}[/red]")
            logging.error(f"Ошибка мониторинга клавиатуры: {e}")
    
    def _handle_hotkey(self, action: str) -> None:
        try:
            if action == 'exit':
                self.console.log("[yellow]CTRL+Q нажато. Завершение работы...[/yellow]")
                logging.info("CTRL+Q нажато -> сигнал остановки.")
                self.stop_signal = True
            elif action == 'pause':
                self.pause_signal = not self.pause_signal
                status = "приостановлен" if self.pause_signal else "возобновлен"
                self.console.log(f"[magenta]Скрипт {status}.[/magenta]")
                logging.info(f"Скрипт {status} пользователем.")
            elif action == 'settings':
                if not self.settings_signal:
                    self.console.log("[cyan]Открытие панели настроек...[/cyan]")
                    logging.info("Открытие панели настроек.")
                    self.settings_signal = True
            elif action == 'debug':
                self.show_debug_window = not self.show_debug_window
                status = "включено" if self.show_debug_window else "отключено"
                self.console.log(f"[cyan]Окно отладки {status}.[/cyan]")
                logging.info(f"Окно отладки {status}.")
                self.save_settings()
            elif action == 'reload':
                self.console.log("[cyan]Перезагрузка модели...[/cyan]")
                try:
                    with self.model_lock:
                        self.model = self.load_model()
                    self.console.log("[green]Модель успешно перезагружена![/green]")
                except Exception as e:
                    self.console.log(f"[red]Ошибка при перезагрузке модели: {e}[/red]")
            elif action == 'find_window':
                self.console.log("[cyan]Поиск окна Telegram...[/cyan]")
                new_window = self.find_telegram_window()
                if new_window:
                    self.window = new_window
                    self.console.log(f"[green]Найдено окно: {self.window.title}[/green]")
                else:
                    self.console.log("[red]Окно Telegram не найдено![/red]")
            elif action == 'toggle_autoplay':
                self.auto_play = not self.auto_play
                status = "включен" if self.auto_play else "отключен"
                self.console.log(f"[cyan]Автоматический запуск игры {status}.[/cyan]")
                self.save_settings()
            elif action == 'toggle_sound':
                self.enable_sound = not self.enable_sound
                status = "включены" if self.enable_sound else "отключены"
                self.console.log(f"[cyan]Звуковые эффекты {status}.[/cyan]")
                self.save_settings()
        except Exception as e:
            self.console.log(f"[red]Ошибка при обработке горячей клавиши '{action}': {e}[/red]")
            logging.error(f"Ошибка при обработке горячей клавиши: {e}")

    def show_settings_panel(self) -> None:
        class SettingsWindow(QtWidgets.QDialog):
            def __init__(self, bot_instance, parent=None):
                super().__init__(parent)
                self.bot_instance = bot_instance
                self.setWindowTitle(f"Настройки бота (v{self.bot_instance.VERSION})")
                self.setFixedSize(550, 650)
                self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
                self.init_ui()

            def init_ui(self):
                main_layout = QtWidgets.QVBoxLayout()
                
                timing_group = QtWidgets.QGroupBox("Задержки и тайминги")
                timing_layout = QtWidgets.QFormLayout()
                
                self.delay_between_clicks_input = QtWidgets.QDoubleSpinBox()
                self.delay_between_clicks_input.setRange(0, 10)
                self.delay_between_clicks_input.setSingleStep(0.01)
                self.delay_between_clicks_input.setDecimals(3)
                self.delay_between_clicks_input.setValue(self.bot_instance.delay_between_clicks)
                self.delay_between_clicks_input.setToolTip("Задержка между последовательными кликами (0 для максимальной скорости)")
                timing_layout.addRow("Задержка между кликами (секунды):", self.delay_between_clicks_input)
                
                self.delay_before_click_input = QtWidgets.QDoubleSpinBox()
                self.delay_before_click_input.setRange(0, 10)
                self.delay_before_click_input.setSingleStep(0.01)
                self.delay_before_click_input.setDecimals(3)
                self.delay_before_click_input.setValue(self.bot_instance.delay_before_click)
                self.delay_before_click_input.setToolTip("Задержка перед выполнением клика (0 для максимальной скорости)")
                timing_layout.addRow("Задержка перед кликом (секунды):", self.delay_before_click_input)
                
                self.fps_lock_input = QtWidgets.QSpinBox()
                self.fps_lock_input.setRange(1, 240)
                self.fps_lock_input.setValue(self.bot_instance.fps_lock)
                self.fps_lock_input.setToolTip("Ограничение частоты кадров для снижения нагрузки на CPU")
                timing_layout.addRow("Ограничение FPS:", self.fps_lock_input)
                
                self.retry_count_input = QtWidgets.QSpinBox()
                self.retry_count_input.setRange(1, 10)
                self.retry_count_input.setValue(self.bot_instance.retry_count)
                self.retry_count_input.setToolTip("Количество попыток повторения операций в случае ошибки")
                timing_layout.addRow("Количество попыток:", self.retry_count_input)
                
                timing_group.setLayout(timing_layout)
                main_layout.addWidget(timing_group)
                
                features_group = QtWidgets.QGroupBox("Функции")
                features_layout = QtWidgets.QVBoxLayout()
                
                self.auto_play_checkbox = QtWidgets.QCheckBox("Автоматический запуск игры")
                self.auto_play_checkbox.setChecked(self.bot_instance.auto_play)
                self.auto_play_checkbox.setToolTip("Автоматически нажимать кнопку 'Играть' когда она появляется")
                features_layout.addWidget(self.auto_play_checkbox)
                
                self.show_debug_window_checkbox = QtWidgets.QCheckBox("Показывать окно отладки")
                self.show_debug_window_checkbox.setChecked(self.bot_instance.show_debug_window)
                self.show_debug_window_checkbox.setToolTip("Отображать окно с визуализацией распознавания")
                features_layout.addWidget(self.show_debug_window_checkbox)
                
                self.click_all_bombs_checkbox = QtWidgets.QCheckBox("Кликать по всем бомбам")
                self.click_all_bombs_checkbox.setChecked(self.bot_instance.click_all_bombs)
                self.click_all_bombs_checkbox.setToolTip("Кликать по всем бомбам, а не только по целевому классу")
                features_layout.addWidget(self.click_all_bombs_checkbox)
                
                self.use_cpu_checkbox = QtWidgets.QCheckBox("Использовать только CPU (без GPU)")
                self.use_cpu_checkbox.setChecked(self.bot_instance.use_cpu)
                self.use_cpu_checkbox.setToolTip("Включите если возникают проблемы с GPU или для экономии ресурсов")
                features_layout.addWidget(self.use_cpu_checkbox)
                
                self.enable_sound_checkbox = QtWidgets.QCheckBox("Включить звуковые эффекты")
                self.enable_sound_checkbox.setChecked(self.bot_instance.enable_sound)
                self.enable_sound_checkbox.setToolTip("Воспроизводить звук при клике")
                features_layout.addWidget(self.enable_sound_checkbox)
                
                features_group.setLayout(features_layout)
                main_layout.addWidget(features_group)
                
                model_group = QtWidgets.QGroupBox("Модель")
                model_layout = QtWidgets.QVBoxLayout()
                
                model_path_layout = QtWidgets.QHBoxLayout()
                self.model_path_input = QtWidgets.QLineEdit(self.bot_instance.model_path)
                self.model_path_input.setToolTip("Путь к файлу модели .pt")
                self.browse_button = QtWidgets.QPushButton("Обзор")
                self.browse_button.clicked.connect(self.browse_file)
                model_path_layout.addWidget(self.model_path_input)
                model_path_layout.addWidget(self.browse_button)
                
                model_layout.addLayout(model_path_layout)
                
                self.reload_model_button = QtWidgets.QPushButton("Перезагрузить модель")
                self.reload_model_button.clicked.connect(self.reload_model)
                model_layout.addWidget(self.reload_model_button)
                
                model_group.setLayout(model_layout)
                main_layout.addWidget(model_group)
                
                info_group = QtWidgets.QGroupBox("Информация о системе")
                info_layout = QtWidgets.QVBoxLayout()
                
                system_info = self.bot_instance.get_system_info()
                info_text = ""
                for key, value in system_info.items():
                    info_text += f"{key}: {value}\n"
                    
                self.info_label = QtWidgets.QLabel(info_text)
                info_layout.addWidget(self.info_label)
                
                last_errors_group = QtWidgets.QGroupBox("Последние ошибки")
                last_errors_layout = QtWidgets.QVBoxLayout()
                
                if self.bot_instance.last_errors:
                    error_text = "\n".join(self.bot_instance.last_errors)
                else:
                    error_text = "Нет ошибок"
                
                self.error_label = QtWidgets.QLabel(error_text)
                self.error_label.setStyleSheet("color: red;")
                last_errors_layout.addWidget(self.error_label)
                
                last_errors_group.setLayout(last_errors_layout)
                info_layout.addWidget(last_errors_group)
                
                info_group.setLayout(info_layout)
                main_layout.addWidget(info_group)
                
                self.button_box = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Save | 
                    QtWidgets.QDialogButtonBox.Cancel |
                    QtWidgets.QDialogButtonBox.Reset
                )
                self.button_box.accepted.connect(self.save_and_close)
                self.button_box.rejected.connect(self.cancel_and_close)
                self.button_box.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.reset_settings)
                main_layout.addWidget(self.button_box)
                
                hotkeys_label = QtWidgets.QLabel(
                    "Горячие клавиши:\n"
                    "CTRL+Q - Выход\n"
                    "CTRL+X - Пауза/Продолжить\n"
                    "CTRL+W - Настройки\n"
                    "CTRL+D - Окно отладки\n"
                    "CTRL+R - Перезагрузить модель\n"
                    "CTRL+F - Найти окно Telegram\n"
                    "CTRL+A - Вкл/Выкл автозапуск\n"
                    "CTRL+S - Вкл/Выкл звук"
                )
                hotkeys_label.setStyleSheet("color: gray;")
                main_layout.addWidget(hotkeys_label)
                
                self.setLayout(main_layout)

            def browse_file(self):
                filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Выбор файла модели", "", "PT файлы (*.pt)"
                )
                if filename:
                    self.model_path_input.setText(filename)

            def reload_model(self):
                try:
                    new_path = self.model_path_input.text()
                    if not os.path.exists(new_path):
                        QtWidgets.QMessageBox.warning(
                            self, "Ошибка", f"Файл модели не найден: {new_path}"
                        )
                        return
                        
                    self.bot_instance.model_path = new_path
                    self.bot_instance.use_cpu = self.use_cpu_checkbox.isChecked()
                    with self.bot_instance.model_lock:
                        self.bot_instance.model = self.bot_instance.load_model()
                        
                    QtWidgets.QMessageBox.information(
                        self, "Успех", "Модель успешно перезагружена!"
                    )
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, "Ошибка", f"Не удалось перезагрузить модель: {e}"
                    )

            def save_and_close(self):
                self.bot_instance.delay_between_clicks = self.delay_between_clicks_input.value()
                self.bot_instance.delay_before_click = self.delay_before_click_input.value()
                self.bot_instance.fps_lock = self.fps_lock_input.value()
                self.bot_instance.auto_play = self.auto_play_checkbox.isChecked()
                self.bot_instance.show_debug_window = self.show_debug_window_checkbox.isChecked()
                self.bot_instance.click_all_bombs = self.click_all_bombs_checkbox.isChecked()
                self.bot_instance.model_path = self.model_path_input.text()
                self.bot_instance.retry_count = self.retry_count_input.value()
                self.bot_instance.use_cpu = self.use_cpu_checkbox.isChecked()
                self.bot_instance.enable_sound = self.enable_sound_checkbox.isChecked()
                
                self.bot_instance.save_settings()
                self.bot_instance.console.log("[green]Настройки обновлены![/green]")
                logging.info("Настройки обновлены пользователем.")
                self.accept()
                self.bot_instance.settings_signal = False

            def cancel_and_close(self):
                self.reject()
                self.bot_instance.console.log("[yellow]Настройки не изменены.[/yellow]")
                logging.info("Настройки не изменены пользователем.")
                self.bot_instance.settings_signal = False
                
            def reset_settings(self):
                reply = QtWidgets.QMessageBox.question(
                    self, 'Сброс настроек',
                    'Вы уверены, что хотите сбросить все настройки до значений по умолчанию?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No
                )
                
                if reply == QtWidgets.QMessageBox.Yes:
                    self.delay_between_clicks_input.setValue(0.0)
                    self.delay_before_click_input.setValue(0.0)
                    self.fps_lock_input.setValue(60)
                    self.auto_play_checkbox.setChecked(True)
                    self.show_debug_window_checkbox.setChecked(False)
                    self.click_all_bombs_checkbox.setChecked(False)
                    self.retry_count_input.setValue(3)
                    self.use_cpu_checkbox.setChecked(not torch.cuda.is_available())
                    self.enable_sound_checkbox.setChecked(True)

        try:
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([])
                
            settings_window = SettingsWindow(bot_instance=self)
            settings_window.exec_()
        except Exception as e:
            self.console.log(f"[red]Ошибка при открытии панели настроек: {e}[/red]")
            logging.error(f"Ошибка при открытии панели настроек: {e}")
            self.settings_signal = False

    def run(self) -> None:
        if self.is_running:
            self.console.log("[yellow]Бот уже запущен.[/yellow]")
            return

        self.is_running = True
        
        try:
            if not self.bring_window_to_foreground():
                self.console.log("[yellow]Не удалось вывести окно на передний план. Продолжаем работу.[/yellow]")
            
            capture_gen = self.capture_telegram_window()
            frame_count = 0
            last_frame_time = time.time()
            results_queue = queue.Queue()
            ocr_thread = None
            fps_history = []
            avg_fps = 0
            
            with Live(console=self.console, refresh_per_second=self.fps_lock) as live:
                while not self.stop_signal:
                    if self.settings_signal:
                        self.show_settings_panel()
                        continue
                    
                    if self.pause_signal:
                        time.sleep(0.1)
                        continue
                    
                    start_time = time.time()
                    
                    try:
                        image, bbox = next(capture_gen)
                    except StopIteration:
                        self.console.log("[yellow]Захват экрана прекращен. Переподключение...[/yellow]")
                        capture_gen = self.capture_telegram_window()
                        time.sleep(0.5)
                        continue
                    except Exception as e:
                        self.console.log(f"[red]Ошибка при получении кадра: {e}[/red]")
                        time.sleep(0.5)
                        continue
                    
                    if frame_count % self.FRAME_SKIP == 0:
                        try:
                            preprocessed_frame = self.preprocess_image(image)
                            with self.model_lock:
                                with torch.no_grad():
                                    prediction = self.model(preprocessed_frame)
                        except Exception as e:
                            self.console.log(f"[red]Ошибка при обработке кадра моделью: {e}[/red]")
                            logging.error(f"Ошибка обработки кадра: {e}")
                            prediction = []
                    else:
                        prediction = []
                    
                    if self.auto_play and (ocr_thread is None or not ocr_thread.is_alive()):
                        ocr_thread = threading.Thread(
                            target=self.detect_play_button, 
                            args=(image, bbox, results_queue), 
                            daemon=True
                        )
                        ocr_thread.start()
                    
                    try:
                        play_button_coords = results_queue.get_nowait()
                        if play_button_coords:
                            x, y = play_button_coords
                            absolute_x = x + bbox["left"]
                            absolute_y = y + bbox["top"]
                            
                            self.console.log(f"[green]Нажимаем кнопку Play по координатам ({absolute_x}, {absolute_y})[/green]")
                            self.perform_click(absolute_x, absolute_y)
                            if self.delay_between_clicks > 0:
                                time.sleep(self.delay_between_clicks)
                    except queue.Empty:
                        pass
                    
                    if prediction:
                        det = prediction[0]
                        if hasattr(det, 'boxes') and det.boxes is not None:
                            try:
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
                                        
                                        x1 = int(x1 * width_scale)
                                        y1 = int(y1 * height_scale)
                                        x2 = int(x2 * width_scale)
                                        y2 = int(y2 * height_scale)
                                        
                                        color = (0, 255, 0) if class_id == self.TARGET_ID else (255, 0, 0)
                                        
                                        cv2.rectangle(self.debug_image, (x1, y1), (x2, y2), color, 2)
                                        
                                        label = f"ID: {class_id}, {score:.2f}"
                                        try:
                                            if hasattr(self.model, 'names') and class_id in self.model.names:
                                                label = f"{self.model.names[class_id]}: {score:.2f}"
                                        except:
                                            pass
                                            
                                        cv2.putText(
                                            self.debug_image, label,
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                                        )
                                        
                                        if class_id == self.TARGET_ID or (self.click_all_bombs and class_id == 0):
                                            x_center = (x1 + x2) // 2 + bbox["left"]
                                            y_center = (y1 + y2) // 2 + bbox["top"]
                                            
                                            if self.delay_before_click > 0:
                                                time.sleep(self.delay_before_click)
                                            
                                            self.last_click_info = self.perform_click(x_center, y_center)
                                            
                                            if self.delay_between_clicks > 0:
                                                time.sleep(self.delay_between_clicks)
                                
                                self.update_debug_window()
                                
                            except Exception as e:
                                self.console.log(f"[red]Ошибка при обработке результатов модели: {e}[/red]")
                                logging.error(f"Ошибка обработки результатов модели: {e}")
                    
                    current_time = time.time()
                    elapsed_time = current_time - last_frame_time
                    fps = 1 / elapsed_time if elapsed_time > 0 else 0
                    
                    fps_history.append(fps)
                    if len(fps_history) > 10:
                        fps_history.pop(0)
                    avg_fps = sum(fps_history) / len(fps_history)
                    
                    last_frame_time = current_time
                    
                    target_frame_time = 1 / self.fps_lock
                    if elapsed_time < target_frame_time:
                        time.sleep(target_frame_time - elapsed_time)
                    
                    system_info = self.get_system_info()
                    
                    if self.last_click_info:
                        x, y, count = self.last_click_info
                        click_info_panel = Panel(
                            f"Последний клик: ({x}, {y})\nКоличество кликов: {count}\n"
                            f"Всего кликов: {sum(self.click_counters.values())}",
                            title="Информация о кликах",
                            border_style="yellow",
                        )
                    else:
                        click_info_panel = Panel(
                            "Кликов еще не было",
                            title="Информация о кликах",
                            border_style="yellow",
                        )
                    
                    system_info_panel = Panel(
                        f"FPS: {avg_fps:.1f} (лимит: {self.fps_lock})\n"
                        f"CPU: {system_info.get('CPU', 'N/A')}\n"
                        f"RAM: {system_info.get('RAM', 'N/A')}\n"
                        f"GPU: {system_info.get('GPU', 'N/A')}\n"
                        f"Всего кликов: {system_info.get('Всего кликов', '0')}\n"
                        f"Скорость: {system_info.get('Скорость', '0')} \n"
                        f"Ошибок: {system_info.get('Ошибок', '0')}",
                        title="Системная информация",
                        border_style="green",
                    )
                    
                    status = "Приостановлено" if self.pause_signal else "Работает"
                    auto_play_status = "Вкл" if self.auto_play else "Выкл"
                    sound_status = "Вкл" if self.enable_sound else "Выкл"
                    debug_status = "Вкл" if self.show_debug_window else "Выкл"
                    
                    hotkeys_panel = Panel(
                        f"Статус: [bold]{status}[/bold]\n"
                        f"Автозапуск: [bold]{auto_play_status}[/bold]\n"
                        f"Звук: [bold]{sound_status}[/bold]\n"
                        f"Отладка: [bold]{debug_status}[/bold]\n"
                        "CTRL+Q - Выход\n"
                        "CTRL+X - Пауза/Продолжить\n"
                        "CTRL+W - Настройки\n"
                        "CTRL+D - Окно отладки\n"
                        "CTRL+F - Найти окно",
                        title=f"Управление (v{self.VERSION})",
                        border_style="magenta",
                    )
                    
                    layout = Layout()
                    layout.split_row(
                        Layout(name="left"),
                        Layout(self.messages_panel, name="right", ratio=2)
                    )
                    layout["left"].split_column(
                        Layout(click_info_panel),
                        Layout(system_info_panel),
                        Layout(hotkeys_panel),
                    )
                    live.update(layout)
                    
                    frame_count += 1
            
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
        except KeyboardInterrupt:
            self.console.log("[yellow]Прервано пользователем.[/yellow]")
            logging.info("Программа прервана пользователем.")
        except Exception as e:
            self.console.log(f"[red]Критическая ошибка: {e}[/red]")
            logging.exception(f"Критическая ошибка: {e}")
            
            error_type = type(e).__name__
            error_trace = traceback.format_exc()
            
            self.console.log(f"[red]Тип ошибки: {error_type}[/red]")
            self.console.log("[red]Трассировка ошибки:[/red]")
            for line in error_trace.split('\n'):
                if line.strip():
                    self.console.log(f"[dim]{line}[/dim]")
            
            if "CUDA" in str(e):
                self.console.log("[yellow]Обнаружена проблема с GPU. Автоматически переключаемся на CPU...[/yellow]")
                self.use_cpu = True
                self.save_settings()
                self.run()
            elif "Permission" in str(e) or "доступ" in str(e).lower():
                self.console.log("[yellow]Проблема с правами доступа. Запустите программу от имени администратора.[/yellow]")
            
        finally:
            self.is_running = False
            self.console.log("[blue]Завершение работы программы...[/blue]")
            logging.info("Программа завершена.")
            
            try:
                with open("stats.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "clicks": sum(self.click_counters.values()),
                        "errors": self.error_count,
                        "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "uptime_seconds": time.time() - self.start_time,
                        "version": self.VERSION
                    }, f, indent=4, ensure_ascii=False)
            except:
                pass

    def __del__(self):
        self.stop_signal = True
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        if hasattr(self, "keyboard_thread") and self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=1.0)


def main():
    print(f"BlumClicker v{BlumClicker.VERSION}")
    print("=" * 40)
    
    if not check_dependencies():
        sys.exit(1)
        
    print("Запуск бота...")
    print("Для выхода нажмите CTRL+Q")
    bot = BlumClicker()
    bot.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print("Для подробностей проверьте файл журнала bot_log.txt")
        logging.critical(f"Необработанное исключение: {e}", exc_info=True)
        traceback.print_exc()
        sys.exit(1)
