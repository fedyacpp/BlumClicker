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
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt
from rich.progress import Progress

CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
WINDOW_TITLE = "Telegram"
STOP_SIGNAL = False
CTRL_Q_PRESSED_ONCE = False
FRAME_SKIP = 1

console = Console()
click_counters = {}


def load_model():
    console.log("[blue]Loading model...[/blue]", highlight=True)
    default_model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    model_path = Prompt.ask("[bold magenta]Path to model weights file[/bold magenta]", default=default_model_path)
    with Progress() as progress:
        task = progress.add_task("[cyan]Loading...", total=100)
        device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = DetectMultiBackend(model_path, device=device)
        model = model.to(device)
        model.warmup(imgsz=(1, 3, 416, 416))
        progress.update(task, advance=100)
    console.log("[green]Model loaded![/green]", highlight=True)
    return model, device


def preprocess_image(image, device):
    image = cv2.resize(image, (416, 416))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).float().div(255.0).unsqueeze(0).to(device)


def find_telegram_window(title_keyword=WINDOW_TITLE):
    console.log(f"[blue]Searching for window '{title_keyword}'...[/blue]", highlight=True)
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        console.log(f"[red]Window '{title_keyword}' not found[/red]", justify="center")
        return None
    console.log(f"[green]Window '{windows[0].title}' found![/green]", highlight=True)
    return windows[0]


def capture_telegram_window(window):
    global STOP_SIGNAL
    hwnd = window._hWnd
    bbox = {'left': window.left, 'top': window.top, 'width': window.width, 'height': window.height}
    with mss.mss() as sct:
        while not STOP_SIGNAL:
            screenshot = sct.grab(bbox)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            yield img


def perform_click(x, y):
    global click_counters
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    if (x, y) not in click_counters:
        click_counters[(x, y)] = 0
    click_counters[(x, y)] += 1

    return (x, y, click_counters[(x, y)])


def check_keyboard():
    global STOP_SIGNAL, CTRL_Q_PRESSED_ONCE
    while not STOP_SIGNAL:
        if keyboard.is_pressed("ctrl+q"):
            if not CTRL_Q_PRESSED_ONCE:
                console.log("[yellow]CTRL+Q. Press again to exit.[/yellow]", highlight=True)
                CTRL_Q_PRESSED_ONCE = True
            else:
                console.log("[green]Exiting...[/green]", highlight=True)
                STOP_SIGNAL = True
        time.sleep(0.1)


def get_system_info():
    cpu_usage = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    mem_usage = mem.percent
    mem_total = mem.total / (1024 ** 3)
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
        "GPU Usage": gpu_usage
    }


def bring_window_to_foreground(window):
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

        console.log(f"[green]Window '{window.title}' brought to the foreground![/green]", highlight=True)
    except Exception as e:
        console.log(f"[red]Error: {e}[/red]", highlight=True)


def main():
    global STOP_SIGNAL
    last_click_info = None

    console.log("[blue]Starting...[/blue]", highlight=True)
    model, device = load_model()
    window = find_telegram_window()
    if not window:
        console.log("[red]Telegram window not found. Exiting...[/red]", highlight=True)
        return

    keyboard_thread = threading.Thread(target=check_keyboard)
    keyboard_thread.start()

    bring_window_to_foreground(window)

    capture_gen = capture_telegram_window(window)
    frame_count = 0

    with Live(console=console, refresh_per_second=10) as live:
        while True:
            if STOP_SIGNAL:
                console.log("[green]Stopping...[/green]", highlight=True)
                break

            start_time = time.time()

            try:
                screenshot = next(capture_gen)
                if frame_count % FRAME_SKIP == 0:
                    preprocessed_frame = preprocess_image(screenshot, device)

                    with torch.no_grad():
                        prediction = model(preprocessed_frame, augment=False, visualize=False)[0]

                    predictions = non_max_suppression(
                        prediction, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD, agnostic=False)
                else:
                    predictions = []

                if predictions and len(predictions) > 0 and predictions[0] is not None:
                    for det in predictions[0]:
                        xyxy = det[:4].tolist()
                        score = det[4].item()

                        if score > 0.5:
                            x1, y1, x2, y2 = xyxy
                            width_scale = (window.width) / 416
                            height_scale = (window.height) / 416
                            x1, y1, x2, y2 = int(x1 * width_scale), int(y1 * height_scale), int(
                                x2 * width_scale), int(y2 * height_scale)

                            x1, y1, x2, y2 = x1 + window.left, y1 + window.top, x2 + window.left, y2 + window.top

                            last_click_info = perform_click((x1 + x2) // 2, (y1 + y2) // 2)

                elapsed_time = time.time() - start_time
                fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')

                system_info = get_system_info()
                if last_click_info:
                    x, y, count = last_click_info
                    click_info_panel = Panel(
                        f"Last click: ({x}, {y})\nClicks: {count}",
                        title="Click Info",
                        border_style="yellow"
                    )
                else:
                    click_info_panel = Panel("No clicks", title="Click Info", border_style="yellow")

                system_info_panel = Panel(
                    f"FPS: {fps:.2f}\n"
                    f"CPU: {system_info['CPU Usage']}\n"
                    f"RAM: {system_info['Memory Usage']}\n"
                    f"RAM Total: {system_info['Total Memory']} GB\n"
                    f"GPU: {system_info['GPU']}\n"
                    f"GPU Usage: {system_info['GPU Usage']}",
                    title="System Info",
                    border_style="green"
                )

                layout = Layout()
                layout.split_column(
                    Layout(click_info_panel),
                    Layout(system_info_panel)
                )

                live.update(layout)

                frame_count += 1

            except StopIteration:
                break
            except Exception as e:
                console.log(f"[red]Error: {e}[/red]")
                STOP_SIGNAL = True

    cv2.destroyAllWindows()
    keyboard_thread.join()
    console.log("[blue]Exiting.[/blue]", highlight=True)


if __name__ == "__main__":
    main()