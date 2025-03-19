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


class Localization:
    LANGUAGES = {
        "ru": "Русский",
        "en": "English",
    }

    STRINGS = {
        "settings_file_not_found": {
            "ru": "Файл настроек не существует. Используются настройки по умолчанию.",
            "en": "Settings file doesn't exist. Using default settings.",
        },
        "settings_loaded": {
            "ru": "Настройки успешно загружены.",
            "en": "Settings successfully loaded.",
        },
        "settings_saved": {
            "ru": "Настройки успешно сохранены.",
            "en": "Settings successfully saved.",
        },
        "settings_updated": {
            "ru": "Настройки обновлены!",
            "en": "Settings updated!",
        },
        "settings_not_changed": {
            "ru": "Настройки не изменены.",
            "en": "Settings not changed.",
        },
        "settings_panel_error": {
            "ru": "Ошибка при открытии панели настроек: {0}",
            "en": "Error opening settings panel: {0}",
        },
        "settings_backup_failed": {
            "ru": "Не удалось создать резервную копию настроек: {0}",
            "en": "Failed to create settings backup: {0}",
        },
        "settings_save_error": {
            "ru": "Ошибка при сохранении настроек: {0}",
            "en": "Error saving settings: {0}",
        },
        "settings_access_error": {
            "ru": "Ошибка доступа: Не удалось сохранить настройки. Проверьте права доступа к файлу {0}",
            "en": "Access error: Could not save settings. Check file permissions for {0}",
        },
        "settings_validation_error": {
            "ru": "Ошибка при конвертации значений в настройках: {0}",
            "en": "Error converting values in settings: {0}",
        },
        "settings_json_error": {
            "ru": "Ошибка: Неверный формат JSON файла. Строка: {0}, Позиция: {1}",
            "en": "Error: Invalid JSON format. Line: {0}, Position: {1}",
        },
        "settings_unexpected_error": {
            "ru": "Непредвиденная ошибка при загрузке настроек: {0}",
            "en": "Unexpected error loading settings: {0}",
        },
        "using_default_settings": {
            "ru": "Используются настройки по умолчанию.",
            "en": "Using default settings.",
        },
        "loading_model": {
            "ru": "Загрузка модели...",
            "en": "Loading model...",
        },
        "model_file_not_found": {
            "ru": "Файл модели не найден по пути: {0}",
            "en": "Model file not found at path: {0}",
        },
        "model_path_prompt": {
            "ru": "Путь к файлу модели",
            "en": "Path to model file",
        },
        "models_found": {
            "ru": "Найдены следующие файлы моделей в текущей директории:",
            "en": "Found the following model files in the current directory:",
        },
        "model_choice_prompt": {
            "ru": "Выберите номер файла или введите полный путь к модели",
            "en": "Select a file number or enter the full path to the model",
        },
        "model_choice_error": {
            "ru": "Ошибка при выборе модели: {0}",
            "en": "Error selecting model: {0}",
        },
        "no_models_found": {
            "ru": "Модели .pt не найдены в текущей директории.",
            "en": ".pt models not found in the current directory.",
        },
        "model_loaded": {
            "ru": "Модель {0} загружена успешно! Обнаружено {1} классов.",
            "en": "Model {0} loaded successfully! Detected {1} classes.",
        },
        "model_classes": {
            "ru": " Классы: {0}",
            "en": " Classes: {0}",
        },
        "model_load_error": {
            "ru": "Не удалось загрузить модель из {0}: {1}",
            "en": "Failed to load model from {0}: {1}",
        },
        "gpu_error": {
            "ru": "Проблема с GPU. Попробуем использовать CPU...",
            "en": "GPU problem. Trying to use CPU...",
        },
        "model_cpu_success": {
            "ru": "Модель успешно загружена на CPU!",
            "en": "Model successfully loaded on CPU!",
        },
        "model_cpu_error": {
            "ru": "Не удалось загрузить модель на CPU: {0}",
            "en": "Failed to load model on CPU: {0}",
        },
        "model_reloaded": {
            "ru": "Модель успешно перезагружена!",
            "en": "Model successfully reloaded!",
        },
        "model_reload_error": {
            "ru": "Ошибка при перезагрузке модели: {0}",
            "en": "Error reloading model: {0}",
        },
        "getting_windows": {
            "ru": "Получение списка всех окон...",
            "en": "Getting list of all windows...",
        },
        "no_visible_windows": {
            "ru": "Не найдено ни одного видимого окна!",
            "en": "No visible windows found!",
        },
        "select_window": {
            "ru": "Выберите окно для захвата:",
            "en": "Select window for capture:",
        },
        "current": {
            "ru": "(текущее)",
            "en": "(current)",
        },
        "window_number_prompt": {
            "ru": "Введите номер окна",
            "en": "Enter window number",
        },
        "invalid_choice": {
            "ru": "Неверный выбор. Используется первое окно из списка.",
            "en": "Invalid choice. Using the first window from the list.",
        },
        "window_selected": {
            "ru": "Выбрано окно: {0}",
            "en": "Window selected: {0}",
        },
        "window_selection_error": {
            "ru": "Ошибка при выборе окна: {0}",
            "en": "Error selecting window: {0}",
        },
        "searching_window": {
            "ru": "Поиск окна...",
            "en": "Searching for window...",
        },
        "saved_window_found": {
            "ru": "Найдено сохраненное окно '{0}'!",
            "en": "Found saved window '{0}'!",
        },
        "saved_window_not_found": {
            "ru": "Сохраненное окно '{0}' не найдено. Выбор нового окна.",
            "en": "Saved window '{0}' not found. Selecting new window.",
        },
        "window_found": {
            "ru": "Найдено окно '{0}'!",
            "en": "Window found '{0}'!",
        },
        "window_not_visible": {
            "ru": "Найдено окно '{0}', но оно может быть не видимым.",
            "en": "Found window '{0}', but it may not be visible.",
        },
        "no_windows_found": {
            "ru": "Окна не найдены. Поиск любого окна...",
            "en": "No windows found. Searching for any window...",
        },
        "window_not_found": {
            "ru": "Окно не найдено. Невозможно вывести на передний план.",
            "en": "Window not found. Cannot bring to foreground.",
        },
        "window_to_foreground": {
            "ru": "Окно '{0}' выведено на передний план!",
            "en": "Window '{0}' brought to foreground!",
        },
        "foreground_attempt": {
            "ru": "Попытка {0}/{1}: Не удалось вывести окно на передний план.",
            "en": "Attempt {0}/{1}: Failed to bring window to foreground.",
        },
        "foreground_error": {
            "ru": "Ошибка при выведении окна на передний план: {0}",
            "en": "Error bringing window to foreground: {0}",
        },
        "foreground_failed": {
            "ru": "Не удалось вывести окно на передний план после нескольких попыток.",
            "en": "Failed to bring window to foreground after several attempts.",
        },
        "capture_error": {
            "ru": "Ошибка при захвате экрана: {0}",
            "en": "Error capturing screen: {0}",
        },
        "window_minimized": {
            "ru": "Окно минимизировано или скрыто. Ожидание...",
            "en": "Window minimized or hidden. Waiting...",
        },
        "invalid_window_size": {
            "ru": "Некорректные размеры окна: {0}x{1}. Ожидание...",
            "en": "Invalid window size: {0}x{1}. Waiting...",
        },
        "too_many_errors": {
            "ru": "Слишком много ошибок подряд. Переподключение к окну...",
            "en": "Too many consecutive errors. Reconnecting to window...",
        },
        "window_reconnected": {
            "ru": "Окно найдено заново!",
            "en": "Window found again!",
        },
        "click_error": {
            "ru": "Ошибка при выполнении клика: {0}",
            "en": "Error performing click: {0}",
        },
        "image_preprocess_error": {
            "ru": "Ошибка при предобработке изображения: {0}",
            "en": "Error preprocessing image: {0}",
        },
        "debug_window_error": {
            "ru": "Ошибка при обновлении окна отладки: {0}",
            "en": "Error updating debug window: {0}",
        },
        "ocr_error": {
            "ru": "Ошибка OCR: {0}",
            "en": "OCR error: {0}",
        },
        "button_detected": {
            "ru": "Кнопка '{0}' обнаружена с уверенностью {1:.2f} по координатам ({2}, {3}, {4}, {5})",
            "en": "Button '{0}' detected with confidence {1:.2f} at coordinates ({2}, {3}, {4}, {5})",
        },
        "button_click": {
            "ru": "Нажимаем кнопку Play по координатам ({0}, {1})",
            "en": "Clicking Play button at coordinates ({0}, {1})",
        },
        "model_results_error": {
            "ru": "Ошибка при обработке результатов модели: {0}",
            "en": "Error processing model results: {0}",
        },
        "init_ocr": {
            "ru": "Инициализация OCR...",
            "en": "Initializing OCR...",
        },
        "using_gpu_ocr": {
            "ru": "Использование GPU для OCR: {0}",
            "en": "Using GPU for OCR: {0}",
        },
        "yes": {
            "ru": "Да",
            "en": "Yes",
        },
        "no": {
            "ru": "Нет",
            "en": "No",
        },
        "ocr_init_success": {
            "ru": "OCR инициализирован успешно!",
            "en": "OCR initialized successfully!",
        },
        "ocr_init_error": {
            "ru": "Ошибка при инициализации OCR: {0}",
            "en": "Error initializing OCR: {0}",
        },
        "hotkeys_registered": {
            "ru": "Горячие клавиши зарегистрированы успешно.",
            "en": "Hotkeys registered successfully.",
        },
        "keyboard_monitor_error": {
            "ru": "Ошибка в мониторинге клавиатуры: {0}",
            "en": "Error in keyboard monitoring: {0}",
        },
        "exit_hotkey": {
            "ru": "CTRL+Q нажато. Завершение работы...",
            "en": "CTRL+Q pressed. Shutting down...",
        },
        "pause_resumed": {
            "ru": "Скрипт {0}.",
            "en": "Script {0}.",
        },
        "paused": {
            "ru": "приостановлен",
            "en": "paused",
        },
        "resumed": {
            "ru": "возобновлен",
            "en": "resumed",
        },
        "opening_settings": {
            "ru": "Открытие панели настроек...",
            "en": "Opening settings panel...",
        },
        "debug_window_toggled": {
            "ru": "Окно отладки {0}.",
            "en": "Debug window {0}.",
        },
        "enabled": {
            "ru": "включено",
            "en": "enabled",
        },
        "disabled": {
            "ru": "отключено",
            "en": "disabled",
        },
        "reloading_model": {
            "ru": "Перезагрузка модели...",
            "en": "Reloading model...",
        },
        "window_selection": {
            "ru": "Выбор окна для захвата...",
            "en": "Selecting window for capture...",
        },
        "window_not_selected": {
            "ru": "Окно не выбрано!",
            "en": "Window not selected!",
        },
        "autoplay_toggled": {
            "ru": "Автоматический запуск игры {0}.",
            "en": "Automatic game start {0}.",
        },
        "sound_toggled": {
            "ru": "Звуковые эффекты {0}.",
            "en": "Sound effects {0}.",
        },
        "hotkey_error": {
            "ru": "Ошибка при обработке горячей клавиши '{0}': {1}",
            "en": "Error processing hotkey '{0}': {1}",
        },
        "language_changed": {
            "ru": "Язык изменен на {0}.",
            "en": "Language changed to {0}.",
        },
        "settings_title": {
            "ru": "Настройки BlumClicker (v{0})",
            "en": "BlumClicker Settings (v{0})",
        },
        "timing_group": {
            "ru": "Задержки и тайминги",
            "en": "Delays and Timing",
        },
        "delay_between_clicks": {
            "ru": "Задержка между кликами (секунды):",
            "en": "Delay between clicks (seconds):",
        },
        "delay_before_click": {
            "ru": "Задержка перед кликом (секунды):",
            "en": "Delay before click (seconds):",
        },
        "fps_lock": {
            "ru": "Ограничение FPS:",
            "en": "FPS Limit:",
        },
        "retry_count": {
            "ru": "Количество попыток:",
            "en": "Retry count:",
        },
        "features_group": {
            "ru": "Функции",
            "en": "Features",
        },
        "auto_play": {
            "ru": "Автоматический запуск игры",
            "en": "Automatic game start",
        },
        "show_debug_window": {
            "ru": "Показывать окно отладки",
            "en": "Show debug window",
        },
        "click_all_bombs": {
            "ru": "Кликать по всем бомбам",
            "en": "Click on all bombs",
        },
        "use_cpu": {
            "ru": "Использовать только CPU (без GPU)",
            "en": "Use CPU only (no GPU)",
        },
        "enable_sound": {
            "ru": "Включить звуковые эффекты",
            "en": "Enable sound effects",
        },
        "current_window": {
            "ru": "Текущее окно:",
            "en": "Current window:",
        },
        "select_window_button": {
            "ru": "Выбрать окно",
            "en": "Select window",
        },
        "model_group": {
            "ru": "Модель",
            "en": "Model",
        },
        "browse_button": {
            "ru": "Обзор",
            "en": "Browse",
        },
        "reload_model_button": {
            "ru": "Перезагрузить модель",
            "en": "Reload model",
        },
        "system_info_group": {
            "ru": "Информация о системе",
            "en": "System information",
        },
        "last_errors_group": {
            "ru": "Последние ошибки",
            "en": "Recent errors",
        },
        "no_errors": {
            "ru": "Нет ошибок",
            "en": "No errors",
        },
        "success": {
            "ru": "Успех",
            "en": "Success",
        },
        "error": {
            "ru": "Ошибка",
            "en": "Error",
        },
        "reset_settings": {
            "ru": "Сброс настроек",
            "en": "Reset settings",
        },
        "reset_confirm": {
            "ru": "Вы уверены, что хотите сбросить все настройки до значений по умолчанию?",
            "en": "Are you sure you want to reset all settings to default values?",
        },
        "select_language": {
            "ru": "Выберите язык:",
            "en": "Select language:",
        },
        "language": {
            "ru": "Язык:",
            "en": "Language:",
        },
        "bot_already_running": {
            "ru": "BlumClicker уже запущен.",
            "en": "BlumClicker is already running.",
        },
        "window_not_found_exit": {
            "ru": "Окно не найдено. Невозможно продолжить работу.",
            "en": "Window not found. Cannot continue.",
        },
        "foreground_continue": {
            "ru": "Не удалось вывести окно на передний план. Продолжаем работу.",
            "en": "Failed to bring window to foreground. Continuing operation.",
        },
        "capture_stopped": {
            "ru": "Захват экрана прекращен. Переподключение...",
            "en": "Screen capture stopped. Reconnecting...",
        },
        "frame_error": {
            "ru": "Ошибка при получении кадра: {0}",
            "en": "Error getting frame: {0}",
        },
        "frame_processing_error": {
            "ru": "Ошибка при обработке кадра моделью: {0}",
            "en": "Error processing frame with model: {0}",
        },
        "interrupted": {
            "ru": "Прервано пользователем.",
            "en": "Interrupted by user.",
        },
        "critical_error": {
            "ru": "Критическая ошибка: {0}",
            "en": "Critical error: {0}",
        },
        "error_type": {
            "ru": "Тип ошибки: {0}",
            "en": "Error type: {0}",
        },
        "error_trace": {
            "ru": "Трассировка ошибки:",
            "en": "Error trace:",
        },
        "gpu_problem": {
            "ru": "Обнаружена проблема с GPU. Автоматически переключаемся на CPU...",
            "en": "GPU problem detected. Automatically switching to CPU...",
        },
        "permission_problem": {
            "ru": "Проблема с правами доступа. Запустите программу от имени администратора.",
            "en": "Permission problem. Run the program as administrator.",
        },
        "program_ending": {
            "ru": "Завершение работы программы...",
            "en": "Program shutdown...",
        },
        "program_terminated": {
            "ru": "Программа завершена.",
            "en": "Program terminated.",
        },
        "messages_title": {
            "ru": "Сообщения",
            "en": "Messages",
        },
        "click_info_title": {
            "ru": "Информация о кликах",
            "en": "Click Information",
        },
        "last_click": {
            "ru": "Последний клик: ({0}, {1})\nКоличество кликов: {2}\nВсего кликов: {3}",
            "en": "Last click: ({0}, {1})\nClick count: {2}\nTotal clicks: {3}",
        },
        "no_clicks": {
            "ru": "Кликов еще не было",
            "en": "No clicks yet",
        },
        "system_info_title": {
            "ru": "Системная информация",
            "en": "System Information",
        },
        "system_info_content": {
            "ru": "FPS: {0:.1f} (лимит: {1})\nCPU: {2}\nRAM: {3}\nGPU: {4}\nВсего кликов: {5}\nСкорость: {6} \nОшибок: {7}",
            "en": "FPS: {0:.1f} (limit: {1})\nCPU: {2}\nRAM: {3}\nGPU: {4}\nTotal clicks: {5}\nSpeed: {6} \nErrors: {7}",
        },
        "controls_title": {
            "ru": "Управление (v{0})",
            "en": "Controls (v{0})",
        },
        "status": {
            "ru": "Статус: [bold]{0}[/bold]",
            "en": "Status: [bold]{0}[/bold]",
        },
        "working": {
            "ru": "Работает",
            "en": "Working",
        },
        "window_short": {
            "ru": "Окно: [bold]{0}...[/bold]",
            "en": "Window: [bold]{0}...[/bold]",
        },
        "autoplay_status": {
            "ru": "Автозапуск: [bold]{0}[/bold]",
            "en": "Auto-start: [bold]{0}[/bold]",
        },
        "sound_status": {
            "ru": "Звук: [bold]{0}[/bold]",
            "en": "Sound: [bold]{0}[/bold]",
        },
        "debug_status": {
            "ru": "Отладка: [bold]{0}[/bold]",
            "en": "Debug: [bold]{0}[/bold]",
        },
        "on": {
            "ru": "Вкл",
            "en": "On",
        },
        "off": {
            "ru": "Выкл",
            "en": "Off",
        },
        "hotkeys": {
            "ru": "CTRL+Q - Выход\nCTRL+X - Пауза/Продолжить\nCTRL+W - Настройки\nCTRL+D - Окно отладки\nCTRL+F - Выбрать окно\nCTRL+L - Сменить язык",
            "en": "CTRL+Q - Exit\nCTRL+X - Pause/Resume\nCTRL+W - Settings\nCTRL+D - Debug window\nCTRL+F - Select window\nCTRL+L - Change language",
        },
        "app_title": {
            "ru": "BlumClicker v{0}",
            "en": "BlumClicker v{0}",
        },
        "exit_info": {
            "ru": "Для выхода нажмите CTRL+Q",
            "en": "Press CTRL+Q to exit",
        },
        "launching_bot": {
            "ru": "Запуск BlumClicker...",
            "en": "Launching BlumClicker...",
        },
        "critical_app_error": {
            "ru": "КРИТИЧЕСКАЯ ОШИБКА: {0}",
            "en": "CRITICAL ERROR: {0}",
        },
        "check_log": {
            "ru": "Для подробностей проверьте файл журнала bot_log.txt",
            "en": "For details check the bot_log.txt log file",
        },
        "missing_packages": {
            "ru": "Отсутствуют необходимые пакеты: {0}",
            "en": "Missing required packages: {0}",
        },
        "install_command": {
            "ru": "Установите их командой:",
            "en": "Install them with the command:",
        },
        "pip_command": {
            "ru": "pip install {0}",
            "en": "pip install {0}",
        },
        "loading_model_progress": {
            "ru": "Загрузка модели...",
            "en": "Loading model...",
        },
        "model_loaded_success": {
            "ru": "Модель загружена успешно!",
            "en": "Model loaded successfully!",
        },
        "model_loaded_classes": {
            "ru": "Модель {0} загружена успешно! Обнаружено {1} классов. Классы: {2}",
            "en": "Model {0} loaded successfully! Detected {1} classes. Classes: {2}",
        },
        "backup_file_created": {
            "ru": "Создана резервная копия файла настроек.",
            "en": "Settings backup file created.",
        },
        "settings_saved_to_file": {
            "ru": "Настройки сохранены в файл.",
            "en": "Settings saved to file.",
        },
        "window_found_specific": {
            "ru": "Найдено окно: {0}",
            "en": "Window found: {0}",
        },
        "window_not_visible_specific": {
            "ru": "Найдено окно (не видимое): {0}",
            "en": "Window found (not visible): {0}",
        },
        "ocr_processing": {
            "ru": "Обработка OCR...",
            "en": "Processing OCR...",
        },
        "button_click_performing": {
            "ru": "Выполнение клика по кнопке...",
            "en": "Performing button click...",
        },
        "error_consecutive_count": {
            "ru": "Последовательных ошибок: {0}",
            "en": "Consecutive errors: {0}",
        },
        "saving_statistics": {
            "ru": "Сохранение статистики...",
            "en": "Saving statistics...",
        },
        "settings_panel_opened": {
            "ru": "Панель настроек открыта.",
            "en": "Settings panel opened.",
        },
        "settings_panel_closed": {
            "ru": "Панель настроек закрыта.",
            "en": "Settings panel closed.",
        },
        "processing_frame": {
            "ru": "Обработка кадра...",
            "en": "Processing frame...",
        },
        "detection_details": {
            "ru": "Обнаружен объект класса {0} с уверенностью {1:.2f} по координатам ({2}, {3}, {4}, {5})",
            "en": "Detected object of class {0} with confidence {1:.2f} at coordinates ({2}, {3}, {4}, {5})",
        },
        "click_performed": {
            "ru": "Выполнен клик по координатам ({0}, {1})",
            "en": "Click performed at coordinates ({0}, {1})",
        },
        "window_capture_init": {
            "ru": "Инициализация захвата окна...",
            "en": "Initializing window capture...",
        },
        "window_captured": {
            "ru": "Захват окна выполнен.",
            "en": "Window captured.",
        },
        "clicking_target": {
            "ru": "Нажатие на цель по координатам ({0}, {1})",
            "en": "Clicking on target at coordinates ({0}, {1})",
        },
        "stats_saved": {
            "ru": "Статистика сохранена.",
            "en": "Statistics saved.",
        },
        "restore_window_focus": {
            "ru": "Восстановление фокуса окна...",
            "en": "Restoring window focus...",
        },
        "error_detail": {
            "ru": "Детали ошибки: {0}",
            "en": "Error details: {0}",
        },
        "script_status": {
            "ru": "Статус скрипта: {0}",
            "en": "Script status: {0}",
        },
        "initialization_complete": {
            "ru": "Инициализация завершена.",
            "en": "Initialization complete.",
        },
        "model_info": {
            "ru": "Информация о модели: {0}",
            "en": "Model info: {0}",
        },
        "initializing_logging": {
            "ru": "Инициализация системы логирования...",
            "en": "Initializing logging system...",
        },
        "logging_initialized": {
            "ru": "Система логирования инициализирована.",
            "en": "Logging system initialized.",
        },
        "settings_backup_created": {
            "ru": "Резервная копия настроек создана.",
            "en": "Settings backup created.",
        },
        "cpu_ram_usage": {
            "ru": "Использование CPU: {0}%, RAM: {1}%",
            "en": "CPU usage: {0}%, RAM: {1}%",
        },
        "gpu_memory_usage": {
            "ru": "Использование GPU: {0}",
            "en": "GPU usage: {0}",
        },
        "bot_uptime": {
            "ru": "Время работы: {0}",
            "en": "Uptime: {0}",
        },
        "click_rate": {
            "ru": "Скорость кликов: {0:.1f} кликов/мин",
            "en": "Click rate: {0:.1f} clicks/min",
        },
        "total_errors": {
            "ru": "Всего ошибок: {0}",
            "en": "Total errors: {0}",
        },
        "settings_file_path": {
            "ru": "Путь к файлу настроек: {0}",
            "en": "Settings file path: {0}",
        },
        "loading_settings": {
            "ru": "Загрузка настроек...",
            "en": "Loading settings...",
        },
        "file_not_found": {
            "ru": "Файл не найден: {0}",
            "en": "File not found: {0}",
        },
        "divider_line": {
            "ru": "=" * 40,
            "en": "=" * 40,
        },
        "version": {
            "ru": "Версия: {0}",
            "en": "Version: {0}",
        },
        "program_started": {
            "ru": "Программа запущена.",
            "en": "Program started.",
        },
        "program_exited": {
            "ru": "Программа завершила работу.",
            "en": "Program exited.",
        },
        "unhandled_exception": {
            "ru": "Необработанное исключение: {0}",
            "en": "Unhandled exception: {0}",
        },
        "scanning_directory": {
            "ru": "Сканирование директории на наличие моделей...",
            "en": "Scanning directory for models...",
        },
        "debug_window_shown": {
            "ru": "Окно отладки показано.",
            "en": "Debug window shown.",
        },
        "debug_window_hidden": {
            "ru": "Окно отладки скрыто.",
            "en": "Debug window hidden.",
        },
        "language_set_to": {
            "ru": "Установлен язык: {0}",
            "en": "Language set to: {0}",
        },
        "no_window": {
            "ru": "Нет окна",
            "en": "No window",
        },
        "window_lost_during_capture": {
            "ru": "Окно потеряно во время захвата.",
            "en": "Window lost during capture.",
        },
        "window_search_again": {
            "ru": "Повторный поиск окна.",
            "en": "Window search again.",
        },
        "invalid_window_handle": {
            "ru": "Неверный дескриптор окна.",
            "en": "Invalid window handle.",
        },
        "loading_window_after_settings": {
            "ru": "После закрытия панели настроек окно не найдено. Поиск нового окна.",
            "en": "Window not found after settings panel closed. Searching for a new window.",
        },
        "recreating_capture_generator": {
            "ru": "Пересоздание генератора захвата окна.",
            "en": "Recreating window capture generator.",
        },
        "window_title_changed": {
            "ru": "Информация о изменении названия окна: {0} -> {1}",
            "en": "Information about window title change: {0} -> {1}",
        },
        "window_will_be_updated": {
            "ru": "Окно будет обновлено при следующем запуске.",
            "en": "Window will be updated on next run.",
        },
        "capture_window_changed": {
            "ru": "Окно изменено во время захвата.",
            "en": "Window changed during capture.",
        },
    }

    def __init__(self, language="ru"):
        self.language = language if language in self.LANGUAGES else "ru"

    def get(self, key, *args):
        if key not in self.STRINGS:
            return f"[Missing: {key}]"

        text = self.STRINGS[key].get(self.language, self.STRINGS[key].get("en", f"[Missing: {key}]"))

        if args:
            try:
                return text.format(*args)
            except Exception as e:
                return f"{text} (format error: {e})"
        return text

    def set_language(self, language):
        if language in self.LANGUAGES:
            self.language = language
            return True
        return False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("bot_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)


def check_dependencies():
    loc = Localization()
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
        "rich": "rich",
    }

    missing_packages = []

    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(loc.get("missing_packages", ", ".join(missing_packages)))
        print(loc.get("install_command"))
        print(loc.get("pip_command", " ".join(missing_packages)))
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
    VERSION = "2.1.0 β"

    def __init__(self) -> None:
        self.stop_signal = False
        self.pause_signal = False
        self.settings_signal = False
        self.is_running = False
        self.window_changed = False

        self.fps_lock = 60
        self.show_debug_window = False
        self.delay_between_clicks = 0.0
        self.delay_before_click = 0.0
        self.auto_play = True
        self.model_path = ""
        self.click_all_bombs = False
        self.retry_count = 3
        self.use_cpu = torch.cuda.is_available() is False
        self.enable_sound = False
        self.selected_window_title = ""
        self.language = "ru"

        self.model_lock = threading.Lock()
        self.ui_lock = threading.Lock()
        self.click_lock = threading.Lock()

        self.click_counters = {}
        self.last_click_info = None
        self.debug_image = None
        self.error_count = {}
        self.last_errors = []
        self.start_time = time.time()

        self.loc = Localization(self.language)

        self.messages_panel = self.MessagesPanel("", title=self.loc.get("messages_title"), border_style="blue")
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
            self.console.log(f"[red]{self.loc.get('critical_error', e)}[/red]")
            logging.error(self.loc.get('critical_error', e))
            print(f"{self.loc.get('critical_app_error', e)}")
            sys.exit(1)

        try:
            self.console.log(f"[blue]{self.loc.get('init_ocr')}[/blue]")
            use_gpu = torch.cuda.is_available() and not self.use_cpu
            self.console.log(f"{self.loc.get('using_gpu_ocr', self.loc.get('yes') if use_gpu else self.loc.get('no'))}")
            self.ocr_reader = easyocr.Reader(["en", "ru"], gpu=use_gpu)
            self.console.log(f"[green]{self.loc.get('ocr_init_success')}[/green]")
        except Exception as e:
            self.console.log(f"[red]{self.loc.get('ocr_init_error', e)}[/red]")
            logging.error(self.loc.get('ocr_init_error', e))
            print(f"{self.loc.get('critical_app_error', e)}")
            self.ocr_reader = None

        self.window = self.find_telegram_window()
        if not self.window:
            self.console.log(f"[red]{self.loc.get('window_not_found_exit')}[/red]")
            logging.error(self.loc.get('window_not_found_exit'))
            print(self.loc.get('window_not_found_exit'))
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
            self.console.log(f"[yellow]{self.loc.get('settings_file_not_found')}[/yellow]")
            logging.warning(self.loc.get('settings_file_not_found'))
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
                self.selected_window_title = settings.get("SELECTED_WINDOW_TITLE", self.selected_window_title)

                language = settings.get("LANGUAGE", self.language)
                if language in Localization.LANGUAGES:
                    self.language = language
                    self.loc.set_language(language)

                model_path = settings.get("MODEL_PATH", self.model_path)
                if model_path and not os.path.exists(model_path):
                    self.console.log(f"[yellow]{self.loc.get('model_file_not_found', model_path)}[/yellow]")
                    logging.warning(self.loc.get('model_file_not_found', model_path))
                else:
                    self.model_path = model_path

                self.show_debug_window = bool(settings.get("SHOW_DEBUG_WINDOW", self.show_debug_window))
                self.click_all_bombs = bool(settings.get("CLICK_ALL_BOMBS", self.click_all_bombs))
                self.retry_count = int(settings.get("RETRY_COUNT", self.retry_count))

                self.console.log(f"[green]{self.loc.get('settings_loaded')}[/green]")
                logging.info(self.loc.get('settings_loaded'))
        except json.JSONDecodeError as e:
            self.console.log(f"[red]{self.loc.get('settings_json_error', e.lineno, e.colno)}[/red]")
            self.console.log(f"[yellow]{self.loc.get('using_default_settings')}[/yellow]")
            logging.warning(self.loc.get('settings_json_error', e.lineno, e.colno))
        except (ValueError, TypeError) as e:
            self.console.log(f"[red]{self.loc.get('settings_validation_error', e)}[/red]")
            self.console.log(f"[yellow]{self.loc.get('using_default_settings')}[/yellow]")
            logging.warning(self.loc.get('settings_validation_error', e))
        except Exception as e:
            self.console.log(f"[red]{self.loc.get('settings_unexpected_error', e)}[/red]")
            self.console.log(f"[yellow]{self.loc.get('using_default_settings')}[/yellow]")
            logging.warning(self.loc.get('settings_unexpected_error', e))

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
            "SELECTED_WINDOW_TITLE": self.selected_window_title,
            "LANGUAGE": self.language,
        }
        try:
            if os.path.exists(self.SETTINGS_FILE):
                backup_file = f"{self.SETTINGS_FILE}.bak"
                try:
                    with open(self.SETTINGS_FILE, "r", encoding="utf-8") as src:
                        with open(backup_file, "w", encoding="utf-8") as dst:
                            dst.write(src.read())
                    logging.info(self.loc.get('backup_file_created'))
                except Exception as e:
                    self.console.log(f"[yellow]{self.loc.get('settings_backup_failed', e)}[/yellow]")
                    logging.warning(self.loc.get('settings_backup_failed', e))

            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as file:
                json.dump(settings, file, indent=4, ensure_ascii=False)
            self.console.log(f"[green]{self.loc.get('settings_saved')}[/green]")
            logging.info(self.loc.get('settings_saved_to_file'))
        except PermissionError:
            self.console.log(f"[red]{self.loc.get('settings_access_error', self.SETTINGS_FILE)}[/red]")
            logging.error(self.loc.get('settings_access_error', self.SETTINGS_FILE))
        except Exception as e:
            self.console.log(f"[red]{self.loc.get('settings_save_error', e)}[/red]")
            logging.error(self.loc.get('settings_save_error', e))

    def load_model(self) -> YOLO:
        self.console.log(f"[blue]{self.loc.get('loading_model')}[/blue]")

        if not self.model_path:
            default_model_path = os.path.join(os.getcwd(), "best.pt")
            self.model_path = Prompt.ask(self.loc.get('model_path_prompt'), default=default_model_path)

        if not os.path.exists(self.model_path):
            self.console.log(f"[red]{self.loc.get('model_file_not_found', self.model_path)}[/red]")
            logging.error(self.loc.get('model_file_not_found', self.model_path))

            model_files = [f for f in os.listdir() if f.endswith('.pt')]
            if model_files:
                self.console.log(f"[yellow]{self.loc.get('models_found')}[/yellow]")
                for i, file in enumerate(model_files):
                    self.console.log(f"  {i+1}. {file}")

                try:
                    choice = Prompt.ask(self.loc.get('model_choice_prompt'), default="1")
                    if choice.isdigit() and 1 <= int(choice) <= len(model_files):
                        self.model_path = model_files[int(choice) - 1]
                    else:
                        self.model_path = choice
                except Exception as e:
                    self.console.log(f"[red]{self.loc.get('model_choice_error', e)}[/red]")
                    raise ValueError(self.loc.get('model_choice_error', e))
            else:
                self.console.log(f"[red]{self.loc.get('no_models_found')}[/red]")
                raise FileNotFoundError(self.loc.get('model_file_not_found', self.model_path))

        try:
            progress = Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{self.loc.get('loading_model_progress')}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )

            with progress:
                task = progress.add_task(f"[cyan]{self.loc.get('loading_model_progress')}", total=100)

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

            try:
                class_names = model.names
                class_count = len(class_names)
                model_info = self.loc.get(
                    'model_loaded_classes',
                    os.path.basename(self.model_path),
                    class_count,
                    ', '.join([f'{i}: {name}' for i, name in class_names.items()]),
                )
            except Exception:
                model_info = self.loc.get('model_loaded', os.path.basename(self.model_path), 0)

            self.console.log(f"[green]{model_info}[/green]")
            logging.info(self.loc.get('model_loaded', self.model_path, 0))
            return model

        except Exception as e:
            error_msg = self.loc.get('model_load_error', self.model_path, e)
            self.console.log(f"[red]{error_msg}[/red]")
            logging.error(error_msg)

            if "CUDA" in str(e) or "cuda" in str(e).lower():
                self.console.log(f"[yellow]{self.loc.get('gpu_error')}[/yellow]")
                try:
                    self.use_cpu = True
                    model = YOLO(self.model_path, task='detect', device='cpu')
                    self.console.log(f"[green]{self.loc.get('model_cpu_success')}[/green]")
                    return model
                except Exception as cpu_error:
                    self.console.log(f"[red]{self.loc.get('model_cpu_error', cpu_error)}[/red]")

            raise RuntimeError(error_msg)

    def select_window(self) -> Optional[gw.Win32Window]:
        self.console.log(f"[blue]{self.loc.get('getting_windows')}[/blue]")

        all_windows = gw.getAllWindows()
        visible_windows = [w for w in all_windows if w.visible and w.title]

        if not visible_windows:
            self.console.log(f"[red]{self.loc.get('no_visible_windows')}[/red]")
            return None

        visible_windows.sort(key=lambda w: w.title.lower())

        self.console.log(f"[green]{self.loc.get('select_window')}[/green]")

        for i, window in enumerate(visible_windows):
            title = window.title
            if title == self.selected_window_title:
                self.console.log(f"[cyan]{i+1}. {title} {self.loc.get('current')}[/cyan]")
            else:
                self.console.log(f"{i+1}. {title}")

        try:
            choice = Prompt.ask(self.loc.get('window_number_prompt'), default="1")
            if choice.isdigit() and 1 <= int(choice) <= len(visible_windows):
                selected_window = visible_windows[int(choice) - 1]
                self.selected_window_title = selected_window.title
                self.save_settings()
                self.console.log(f"[green]{self.loc.get('window_selected', selected_window.title)}[/green]")
                logging.info(self.loc.get('window_selected', selected_window.title))
                return selected_window
            else:
                self.console.log(f"[yellow]{self.loc.get('invalid_choice')}[/yellow]")
                selected_window = visible_windows[0]
                self.selected_window_title = selected_window.title
                self.save_settings()
                return selected_window
        except Exception as e:
            self.console.log(f"[red]{self.loc.get('window_selection_error', e)}[/red]")
            logging.error(self.loc.get('window_selection_error', e))
            return None

    def find_telegram_window(self) -> Optional[gw.Win32Window]:
        self.console.log(f"[blue]{self.loc.get('searching_window')}[/blue]")

        if self.selected_window_title:
            windows = gw.getWindowsWithTitle(self.selected_window_title)
            visible_windows = [w for w in windows if w.visible]

            if visible_windows:
                window = visible_windows[0]
                self.console.log(f"[green]{self.loc.get('saved_window_found', window.title)}[/green]")
                logging.info(self.loc.get('saved_window_found', window.title))
                return window
            else:
                self.console.log(f"[yellow]{self.loc.get('saved_window_not_found', self.selected_window_title)}[/yellow]")

        selected_window = self.select_window()
        if selected_window:
            return selected_window

        for title in self.WINDOW_TITLES:
            windows = gw.getWindowsWithTitle(title)
            visible_windows = [w for w in windows if w.visible]

            if visible_windows:
                window = visible_windows[0]
                self.selected_window_title = window.title
                self.save_settings()
                self.console.log(f"[green]{self.loc.get('window_found', window.title)}[/green]")
                logging.info(self.loc.get('window_found_specific', window.title))
                return window
            elif windows:
                window = windows[0]
                self.selected_window_title = window.title
                self.save_settings()
                self.console.log(f"[yellow]{self.loc.get('window_not_visible', window.title)}[/yellow]")
                logging.info(self.loc.get('window_not_visible_specific', window.title))
                return window

        self.console.log(f"[yellow]{self.loc.get('no_windows_found')}[/yellow]")
        all_windows = gw.getAllWindows()
        for window in all_windows:
            if window.visible and window.title:
                self.selected_window_title = window.title
                self.save_settings()
                self.console.log(f"[green]{self.loc.get('window_found', window.title)}[/green]")
                return window

        return None

    def bring_window_to_foreground(self) -> bool:
        if not self.window:
            self.console.log(f"[red]{self.loc.get('window_not_found')}[/red]")
            if self.selected_window_title:
                windows = gw.getWindowsWithTitle(self.selected_window_title)
                visible_windows = [w for w in windows if w.visible]
                if visible_windows:
                    self.window = visible_windows[0]
                    self.console.log(f"[green]{self.loc.get('window_found', self.window.title)}[/green]")
                else:
                    return False
            else:
                return False

        try:
            hwnd = self.window._hWnd
        except AttributeError:
            self.console.log(f"[red]{self.loc.get('invalid_window_handle')}[/red]")
            self.window = None
            return False

        for attempt in range(self.retry_count):
            try:
                current_foreground = win32gui.GetForegroundWindow()

                if current_foreground != hwnd:
                    window_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
                    current_thread = win32api.GetCurrentThreadId()

                    if window_thread != current_thread:
                        win32process.AttachThreadInput(window_thread, current_thread, True)

                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    else:
                        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

                    win32gui.SetForegroundWindow(hwnd)

                    if window_thread != current_thread:
                        win32process.AttachThreadInput(window_thread, current_thread, False)

                    time.sleep(0.1)

                foreground_hwnd = win32gui.GetForegroundWindow()
                if foreground_hwnd == hwnd:
                    self.console.log(f"[green]{self.loc.get('window_to_foreground', self.window.title)}[/green]")
                    logging.info(self.loc.get('window_to_foreground', self.window.title))
                    return True
                else:
                    self.console.log(f"[yellow]{self.loc.get('foreground_attempt', attempt + 1, self.retry_count)}[/yellow]")
                    time.sleep(0.2)
            except Exception as e:
                self.console.log(f"[red]{self.loc.get('foreground_error', e)}[/red]")
                logging.warning(self.loc.get('foreground_error', e))
                time.sleep(0.2)

        self.console.log(f"[red]{self.loc.get('foreground_failed')}[/red]")
        return False

    def capture_telegram_window(self) -> Generator[Tuple[np.ndarray, Dict[str, int]], None, None]:
        if not self.window:
            self.console.log(f"[red]{self.loc.get('window_not_found')}[/red]")
            time.sleep(0.5)
            if self.selected_window_title:
                windows = gw.getWindowsWithTitle(self.selected_window_title)
                visible_windows = [w for w in windows if w.visible]
                if visible_windows:
                    self.window = visible_windows[0]
                    self.console.log(f"[green]{self.loc.get('window_found', self.window.title)}[/green]")
                else:
                    self.console.log(f"[yellow]{self.loc.get('window_search_again')}[/yellow]")
                    self.window = self.find_telegram_window()
            else:
                self.window = self.find_telegram_window()
                
            if not self.window:
                yield np.zeros((100, 100, 3), dtype=np.uint8), {"left": 0, "top": 0, "width": 100, "height": 100}
                return

        current_window_title = self.window.title
        
        try:
            hwnd = self.window._hWnd
        except AttributeError:
            self.console.log(f"[red]{self.loc.get('invalid_window_handle')}[/red]")
            yield np.zeros((100, 100, 3), dtype=np.uint8), {"left": 0, "top": 0, "width": 100, "height": 100}
            return
            
        last_error_time = 0
        consecutive_errors = 0

        with mss.mss() as sct:
            while not self.stop_signal:
                if self.window_changed or (self.window and current_window_title != self.window.title):
                    self.console.log(f"[yellow]{self.loc.get('capture_window_changed')}[/yellow]")
                    break
                    
                if self.pause_signal or self.settings_signal:
                    time.sleep(0.1)
                    continue
                
                if not self.window:
                    self.console.log(f"[red]{self.loc.get('window_lost_during_capture')}[/red]")
                    yield np.zeros((100, 100, 3), dtype=np.uint8), {"left": 0, "top": 0, "width": 100, "height": 100}
                    return
                    
                try:
                    window_rect = win32gui.GetWindowRect(hwnd)

                    if window_rect[0] < -32000 or window_rect[1] < -32000:
                        if time.time() - last_error_time > 5:
                            self.console.log(f"[yellow]{self.loc.get('window_minimized')}[/yellow]")
                            last_error_time = time.time()
                        time.sleep(0.5)
                        continue

                    width = window_rect[2] - window_rect[0]
                    height = window_rect[3] - window_rect[1]

                    if width <= 0 or height <= 0:
                        if time.time() - last_error_time > 5:
                            self.console.log(f"[yellow]{self.loc.get('invalid_window_size', width, height)}[/yellow]")
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
                        self.console.log(f"[red]{self.loc.get('capture_error', e)}[/red]")
                        logging.error(self.loc.get('capture_error', e))
                        last_error_time = now

                    if consecutive_errors > 5:
                        self.console.log(f"[yellow]{self.loc.get('too_many_errors')}[/yellow]")
                        new_window = self.find_telegram_window()
                        if new_window:
                            self.window = new_window
                            hwnd = self.window._hWnd
                            self.console.log(f"[green]{self.loc.get('window_reconnected')}[/green]")
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
            self.console.log(f"[red]{self.loc.get('click_error', e)}[/red]")
            logging.warning(self.loc.get('click_error', e))
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
            self.console.log(f"[red]{self.loc.get('image_preprocess_error', e)}[/red]")
            logging.error(self.loc.get('image_preprocess_error', e))
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
                        hwnd,
                        win32con.HWND_TOPMOST,
                        0,
                        0,
                        0,
                        0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
                    )
                cv2.waitKey(1)
            except Exception as e:
                self.console.log(f"[yellow]{self.loc.get('debug_window_error', e)}[/yellow]")
                logging.warning(self.loc.get('debug_window_error', e))
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
                            debug_img,
                            f"{text} ({prob:.2f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    self.console.log(f"[green]{self.loc.get('button_detected', text, prob, x, y, w, h)}[/green]")
                    results_queue.put((x + w // 2, y + h // 2))
                    found = True
                    break

            if debug_img is not None and self.show_debug_window:
                cv2.imshow("OCR Debug", debug_img)
                cv2.waitKey(1)

            if not found:
                results_queue.put(None)

        except Exception as e:
            self.console.log(f"[red]{self.loc.get('ocr_error', e)}[/red]")
            logging.warning(self.loc.get('ocr_error', e))
            logging.debug(traceback.format_exc())
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
                    gpu_info = self.loc.get('use_cpu')
                    gpu_usage = "N/A"
                    gpu_temp = "N/A"
            except Exception as e:
                gpu_info = f"{self.loc.get('error')}: {type(e).__name__}"
                gpu_usage = "N/A"
                gpu_temp = "N/A"
                logging.warning(self.loc.get('gpu_problem'))

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
                "Ошибок": f"{sum(self.error_count.values())}",
            }
        except Exception as e:
            logging.error(self.loc.get('error_detail', e))
            return {"Ошибка": self.loc.get('error_detail', e)}

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
            keyboard.add_hotkey('ctrl+l', lambda: self._handle_hotkey('toggle_language'))

            self.console.log(f"[green]{self.loc.get('hotkeys_registered')}[/green]")

            while not self.stop_signal:
                time.sleep(0.1)

        except Exception as e:
            self.console.log(f"[red]{self.loc.get('keyboard_monitor_error', e)}[/red]")
            logging.error(self.loc.get('keyboard_monitor_error', e))

    def _handle_hotkey(self, action: str) -> None:
        try:
            if action == 'exit':
                self.console.log(f"[yellow]{self.loc.get('exit_hotkey')}[/yellow]")
                logging.info(self.loc.get('exit_hotkey'))
                self.stop_signal = True
            elif action == 'pause':
                self.pause_signal = not self.pause_signal
                status = self.loc.get('paused') if self.pause_signal else self.loc.get('resumed')
                self.console.log(f"[magenta]{self.loc.get('pause_resumed', status)}[/magenta]")
                logging.info(self.loc.get('pause_resumed', status))
            elif action == 'settings':
                if not self.settings_signal:
                    self.console.log(f"[cyan]{self.loc.get('opening_settings')}[/cyan]")
                    logging.info(self.loc.get('opening_settings'))
                    self.settings_signal = True
            elif action == 'debug':
                self.show_debug_window = not self.show_debug_window
                status = self.loc.get('enabled') if self.show_debug_window else self.loc.get('disabled')
                self.console.log(f"[cyan]{self.loc.get('debug_window_toggled', status)}[/cyan]")
                logging.info(self.loc.get('debug_window_toggled', status))
                self.save_settings()
            elif action == 'reload':
                self.console.log(f"[cyan]{self.loc.get('reloading_model')}[/cyan]")
                try:
                    with self.model_lock:
                        self.model = self.load_model()
                    self.console.log(f"[green]{self.loc.get('model_reloaded')}[/green]")
                except Exception as e:
                    self.console.log(f"[red]{self.loc.get('model_reload_error', e)}[/red]")
            elif action == 'find_window':
                self.console.log(f"[cyan]{self.loc.get('window_selection')}[/cyan]")
                new_window = self.select_window()
                if new_window:
                    self.window = new_window
                    self.console.log(f"[green]{self.loc.get('window_selected', self.window.title)}[/green]")
                else:
                    self.console.log(f"[red]{self.loc.get('window_not_selected')}[/red]")
            elif action == 'toggle_autoplay':
                self.auto_play = not self.auto_play
                status = self.loc.get('enabled') if self.auto_play else self.loc.get('disabled')
                self.console.log(f"[cyan]{self.loc.get('autoplay_toggled', status)}[/cyan]")
                self.save_settings()
            elif action == 'toggle_sound':
                self.enable_sound = not self.enable_sound
                status = self.loc.get('enabled') if self.enable_sound else self.loc.get('disabled')
                self.console.log(f"[cyan]{self.loc.get('sound_toggled', status)}[/cyan]")
                self.save_settings()
            elif action == 'toggle_language':
                new_lang = "en" if self.language == "ru" else "ru"
                self.language = new_lang
                self.loc.set_language(new_lang)
                self.console.log(f"[cyan]{self.loc.get('language_changed', self.loc.LANGUAGES[new_lang])}[/cyan]")
                self.save_settings()
        except Exception as e:
            self.console.log(f"[red]{self.loc.get('hotkey_error', action, e)}[/red]")
            logging.error(self.loc.get('hotkey_error', action, e))

    def show_settings_panel(self) -> None:
        class SettingsWindow(QtWidgets.QDialog):
            def __init__(self, bot_instance, parent=None):
                super().__init__(parent)
                self.bot_instance = bot_instance
                self.setWindowTitle(self.bot_instance.loc.get('settings_title', self.bot_instance.VERSION))
                self.setFixedSize(550, 750)
                self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
                self.init_ui()

            def init_ui(self):
                main_layout = QtWidgets.QVBoxLayout()
                main_layout.setSpacing(10)
                
                scroll_area = QtWidgets.QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
                
                scroll_content = QtWidgets.QWidget()
                scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
                scroll_layout.setSpacing(10)

                timing_group = QtWidgets.QGroupBox(self.bot_instance.loc.get("timing_group"))
                timing_layout = QtWidgets.QFormLayout()

                self.delay_between_clicks_input = QtWidgets.QDoubleSpinBox()
                self.delay_between_clicks_input.setRange(0, 10)
                self.delay_between_clicks_input.setSingleStep(0.01)
                self.delay_between_clicks_input.setDecimals(3)
                self.delay_between_clicks_input.setValue(self.bot_instance.delay_between_clicks)
                self.delay_between_clicks_input.setToolTip(self.bot_instance.loc.get("delay_between_clicks"))
                timing_layout.addRow(self.bot_instance.loc.get('delay_between_clicks'), self.delay_between_clicks_input)

                self.delay_before_click_input = QtWidgets.QDoubleSpinBox()
                self.delay_before_click_input.setRange(0, 10)
                self.delay_before_click_input.setSingleStep(0.01)
                self.delay_before_click_input.setDecimals(3)
                self.delay_before_click_input.setValue(self.bot_instance.delay_before_click)
                self.delay_before_click_input.setToolTip(self.bot_instance.loc.get("delay_before_click"))
                timing_layout.addRow(self.bot_instance.loc.get('delay_before_click'), self.delay_before_click_input)

                self.fps_lock_input = QtWidgets.QSpinBox()
                self.fps_lock_input.setRange(1, 240)
                self.fps_lock_input.setValue(self.bot_instance.fps_lock)
                self.fps_lock_input.setToolTip(self.bot_instance.loc.get("fps_lock"))
                timing_layout.addRow(self.bot_instance.loc.get('fps_lock'), self.fps_lock_input)

                self.retry_count_input = QtWidgets.QSpinBox()
                self.retry_count_input.setRange(1, 10)
                self.retry_count_input.setValue(self.bot_instance.retry_count)
                self.retry_count_input.setToolTip(self.bot_instance.loc.get("retry_count"))
                timing_layout.addRow(self.bot_instance.loc.get('retry_count'), self.retry_count_input)

                timing_group.setLayout(timing_layout)
                scroll_layout.addWidget(timing_group)

                features_group = QtWidgets.QGroupBox(self.bot_instance.loc.get("features_group"))
                features_layout = QtWidgets.QVBoxLayout()

                self.auto_play_checkbox = QtWidgets.QCheckBox(self.bot_instance.loc.get("auto_play"))
                self.auto_play_checkbox.setChecked(self.bot_instance.auto_play)
                self.auto_play_checkbox.setToolTip(self.bot_instance.loc.get("auto_play"))
                features_layout.addWidget(self.auto_play_checkbox)

                self.show_debug_window_checkbox = QtWidgets.QCheckBox(self.bot_instance.loc.get("show_debug_window"))
                self.show_debug_window_checkbox.setChecked(self.bot_instance.show_debug_window)
                self.show_debug_window_checkbox.setToolTip(self.bot_instance.loc.get("show_debug_window"))
                features_layout.addWidget(self.show_debug_window_checkbox)

                self.click_all_bombs_checkbox = QtWidgets.QCheckBox(self.bot_instance.loc.get("click_all_bombs"))
                self.click_all_bombs_checkbox.setChecked(self.bot_instance.click_all_bombs)
                self.click_all_bombs_checkbox.setToolTip(self.bot_instance.loc.get("click_all_bombs"))
                features_layout.addWidget(self.click_all_bombs_checkbox)

                self.use_cpu_checkbox = QtWidgets.QCheckBox(self.bot_instance.loc.get("use_cpu"))
                self.use_cpu_checkbox.setChecked(self.bot_instance.use_cpu)
                self.use_cpu_checkbox.setToolTip(self.bot_instance.loc.get("use_cpu"))
                features_layout.addWidget(self.use_cpu_checkbox)

                self.enable_sound_checkbox = QtWidgets.QCheckBox(self.bot_instance.loc.get("enable_sound"))
                self.enable_sound_checkbox.setChecked(self.bot_instance.enable_sound)
                self.enable_sound_checkbox.setToolTip(self.bot_instance.loc.get("enable_sound"))
                features_layout.addWidget(self.enable_sound_checkbox)

                language_group = QtWidgets.QGroupBox(self.bot_instance.loc.get("language"))
                language_layout = QtWidgets.QHBoxLayout()
                language_layout.setContentsMargins(10, 10, 10, 10)

                self.language_combo = QtWidgets.QComboBox()
                for lang_code, lang_name in Localization.LANGUAGES.items():
                    self.language_combo.addItem(lang_name, lang_code)
                    if lang_code == self.bot_instance.language:
                        self.language_combo.setCurrentText(lang_name)

                language_layout.addWidget(self.language_combo)
                language_group.setLayout(language_layout)
                features_layout.addWidget(language_group)

                window_group = QtWidgets.QGroupBox(self.bot_instance.loc.get("current_window"))
                window_layout = QtWidgets.QHBoxLayout()
                window_layout.setContentsMargins(10, 10, 10, 10)

                self.window_text = QtWidgets.QLineEdit(self.bot_instance.selected_window_title)
                self.window_text.setReadOnly(True)
                self.select_window_button = QtWidgets.QPushButton(self.bot_instance.loc.get("select_window_button"))
                self.select_window_button.clicked.connect(self.select_window)

                window_layout.addWidget(self.window_text, 3)
                window_layout.addWidget(self.select_window_button, 1)
                window_group.setLayout(window_layout)
                features_layout.addWidget(window_group)

                features_group.setLayout(features_layout)
                scroll_layout.addWidget(features_group)

                model_group = QtWidgets.QGroupBox(self.bot_instance.loc.get("model_group"))
                model_layout = QtWidgets.QVBoxLayout()

                model_path_layout = QtWidgets.QHBoxLayout()
                self.model_path_input = QtWidgets.QLineEdit(self.bot_instance.model_path)
                self.model_path_input.setToolTip(self.bot_instance.loc.get("model_path_prompt"))
                self.browse_button = QtWidgets.QPushButton(self.bot_instance.loc.get("browse_button"))
                self.browse_button.clicked.connect(self.browse_file)
                model_path_layout.addWidget(self.model_path_input)
                model_path_layout.addWidget(self.browse_button)

                model_layout.addLayout(model_path_layout)

                self.reload_model_button = QtWidgets.QPushButton(self.bot_instance.loc.get("reload_model_button"))
                self.reload_model_button.clicked.connect(self.reload_model)
                model_layout.addWidget(self.reload_model_button)

                model_group.setLayout(model_layout)
                scroll_layout.addWidget(model_group)

                info_group = QtWidgets.QGroupBox(self.bot_instance.loc.get("system_info_group"))
                info_layout = QtWidgets.QVBoxLayout()

                system_info = self.bot_instance.get_system_info()
                info_text = ""
                for key, value in system_info.items():
                    info_text += f"{key}: {value}\n"

                self.info_label = QtWidgets.QLabel(info_text)
                info_layout.addWidget(self.info_label)

                last_errors_group = QtWidgets.QGroupBox(self.bot_instance.loc.get("last_errors_group"))
                last_errors_layout = QtWidgets.QVBoxLayout()

                if self.bot_instance.last_errors:
                    error_text = "\n".join(self.bot_instance.last_errors)
                else:
                    error_text = self.bot_instance.loc.get("no_errors")

                self.error_label = QtWidgets.QLabel(error_text)
                self.error_label.setStyleSheet("color: red;")
                last_errors_layout.addWidget(self.error_label)

                last_errors_group.setLayout(last_errors_layout)
                info_layout.addWidget(last_errors_group)

                info_group.setLayout(info_layout)
                scroll_layout.addWidget(info_group)

                scroll_area.setWidget(scroll_content)
                main_layout.addWidget(scroll_area)

                self.button_box = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Save
                    | QtWidgets.QDialogButtonBox.Cancel
                    | QtWidgets.QDialogButtonBox.Reset
                )
                self.button_box.accepted.connect(self.save_and_close)
                self.button_box.rejected.connect(self.cancel_and_close)
                self.button_box.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(self.reset_settings)
                main_layout.addWidget(self.button_box)

                hotkeys_label = QtWidgets.QLabel(self.bot_instance.loc.get("hotkeys"))
                hotkeys_label.setStyleSheet("color: gray;")
                main_layout.addWidget(hotkeys_label)

                self.setLayout(main_layout)

            def browse_file(self):
                filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, self.bot_instance.loc.get("model_path_prompt"), "", "PT files (*.pt)"
                )
                if filename:
                    self.model_path_input.setText(filename)

            def reload_model(self):
                try:
                    new_path = self.model_path_input.text()
                    if not os.path.exists(new_path):
                        QtWidgets.QMessageBox.warning(
                            self,
                            self.bot_instance.loc.get("error"),
                            self.bot_instance.loc.get('model_file_not_found', new_path),
                        )
                        return

                    self.bot_instance.model_path = new_path
                    self.bot_instance.use_cpu = self.use_cpu_checkbox.isChecked()
                    with self.bot_instance.model_lock:
                        self.bot_instance.model = self.bot_instance.load_model()

                    QtWidgets.QMessageBox.information(
                        self, self.bot_instance.loc.get("success"), self.bot_instance.loc.get("model_reloaded")
                    )
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, self.bot_instance.loc.get("error"), self.bot_instance.loc.get('model_reload_error', e)
                    )

            def select_window(self):
                try:
                    class WindowSelectionDialog(QtWidgets.QDialog):
                        def __init__(self, parent=None, bot_instance=None):
                            super().__init__(parent)
                            self.bot_instance = bot_instance
                            self.selected_window = None
                            self.setWindowTitle(self.bot_instance.loc.get('select_window'))
                            self.setMinimumWidth(400)
                            self.init_ui()
                            
                        def init_ui(self):
                            layout = QtWidgets.QVBoxLayout()
                            
                            label = QtWidgets.QLabel(self.bot_instance.loc.get('window_number_prompt'))
                            layout.addWidget(label)
                            
                            all_windows = gw.getAllWindows()
                            visible_windows = [w for w in all_windows if w.visible and w.title]
                            visible_windows.sort(key=lambda w: w.title.lower())
                            
                            if not visible_windows:
                                error_label = QtWidgets.QLabel(self.bot_instance.loc.get('no_visible_windows'))
                                error_label.setStyleSheet("color: red;")
                                layout.addWidget(error_label)
                            else:
                                self.windows_list = QtWidgets.QListWidget()
                                
                                for i, window in enumerate(visible_windows):
                                    title = window.title
                                    item = QtWidgets.QListWidgetItem(title)
                                    item.setData(QtCore.Qt.UserRole, window)
                                    self.windows_list.addItem(item)
                                    
                                    if title == self.bot_instance.selected_window_title:
                                        self.windows_list.setCurrentItem(item)
                                
                                layout.addWidget(self.windows_list)
                                
                            button_box = QtWidgets.QDialogButtonBox(
                                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                            )
                            button_box.accepted.connect(self.accept_selection)
                            button_box.rejected.connect(self.reject)
                            layout.addWidget(button_box)
                            
                            self.setLayout(layout)
                            
                        def accept_selection(self):
                            current_item = self.windows_list.currentItem()
                            if current_item:
                                self.selected_window = current_item.data(QtCore.Qt.UserRole)
                                self.accept()
                            else:
                                self.reject()
                    
                    select_dialog = WindowSelectionDialog(self, self.bot_instance)
                    if select_dialog.exec_() == QtWidgets.QDialog.Accepted and select_dialog.selected_window:
                        new_window = select_dialog.selected_window
                        old_window_title = self.bot_instance.selected_window_title
                        self.bot_instance.window = new_window
                        self.bot_instance.selected_window_title = new_window.title
                        
                        if old_window_title != new_window.title:
                            self.bot_instance.window_changed = True
                            self.bot_instance.console.log(f"[yellow]{self.bot_instance.loc.get('window_title_changed', old_window_title, new_window.title)}[/yellow]")
                            
                        self.bot_instance.save_settings()
                        self.window_text.setText(self.bot_instance.selected_window_title)
                        if not self.bot_instance.bring_window_to_foreground():
                            self.bot_instance.window = None
                            self.bot_instance.console.log(f"[yellow]{self.bot_instance.loc.get('window_will_reinitialize')}[/yellow]")
                        QtWidgets.QMessageBox.information(
                            self,
                            self.bot_instance.loc.get("success"),
                            self.bot_instance.loc.get('window_selected', self.bot_instance.selected_window_title),
                        )
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, self.bot_instance.loc.get("error"), self.bot_instance.loc.get('window_selection_error', e)
                    )

            def save_and_close(self):
                window_changed = False
                old_window_title = self.bot_instance.selected_window_title
                
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

                if old_window_title != self.window_text.text():
                    window_changed = True
                    self.bot_instance.console.log(f"[yellow]{self.bot_instance.loc.get('window_title_changed', old_window_title, self.window_text.text())}[/yellow]")

                try:
                    selected_lang_code = self.language_combo.currentData()
                    if selected_lang_code and selected_lang_code != self.bot_instance.language:
                        if selected_lang_code in Localization.LANGUAGES:
                            self.bot_instance.language = selected_lang_code
                            self.bot_instance.loc.set_language(selected_lang_code)
                except Exception as e:
                    self.bot_instance.console.log(f"[yellow]{self.bot_instance.loc.get('error_detail', e)}[/yellow]")

                self.bot_instance.save_settings()
                if window_changed:
                    self.bot_instance.window = None
                    self.bot_instance.window_changed = True
                    self.bot_instance.console.log(f"[yellow]{self.bot_instance.loc.get('window_will_be_updated')}[/yellow]")
                
                self.bot_instance.console.log(f"[green]{self.bot_instance.loc.get('settings_updated')}[/green]")
                logging.info(self.bot_instance.loc.get('settings_updated'))
                self.accept()
                self.bot_instance.settings_signal = False

                self.bot_instance.bring_window_to_foreground()

            def cancel_and_close(self):
                self.reject()
                self.bot_instance.console.log(f"[yellow]{self.bot_instance.loc.get('settings_not_changed')}[/yellow]")
                logging.info(self.bot_instance.loc.get('settings_not_changed'))
                self.bot_instance.settings_signal = False

                self.bot_instance.bring_window_to_foreground()

            def reset_settings(self):
                reply = QtWidgets.QMessageBox.question(
                    self,
                    self.bot_instance.loc.get('reset_settings'),
                    self.bot_instance.loc.get('reset_confirm'),
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
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

            time.sleep(0.5)
            
            if self.window is None and self.selected_window_title:
                self.console.log(f"[blue]{self.loc.get('loading_window_after_settings')}[/blue]")
                windows = gw.getWindowsWithTitle(self.selected_window_title)
                visible_windows = [w for w in windows if w.visible]
                if visible_windows:
                    self.window = visible_windows[0]
                    self.console.log(f"[green]{self.loc.get('saved_window_found', self.window.title)}[/green]")
                    logging.info(self.loc.get('saved_window_found', self.window.title))
                else:
                    self.window = self.find_telegram_window()
            
            if self.window and not self.bring_window_to_foreground():
                self.console.log(f"[yellow]{self.loc.get('foreground_continue')}[/yellow]")

            self.console.log(f"[cyan]{self.loc.get('restore_window_focus')}[/cyan]")

        except Exception as e:
            self.console.log(f"[red]{self.loc.get('settings_panel_error', e)}[/red]")
            logging.error(self.loc.get('settings_panel_error', e))
            self.settings_signal = False
            
            if self.window is None and self.selected_window_title:
                windows = gw.getWindowsWithTitle(self.selected_window_title)
                visible_windows = [w for w in windows if w.visible]
                if visible_windows:
                    self.window = visible_windows[0]
                    self.console.log(f"[green]{self.loc.get('saved_window_found', self.window.title)}[/green]")
                else:
                    self.window = self.find_telegram_window()
            
            if self.window:
                self.bring_window_to_foreground()

    def run(self) -> None:
        if self.is_running:
            self.console.log(f"[yellow]{self.loc.get('bot_already_running')}[/yellow]")
            return

        self.is_running = True

        try:
            if self.selected_window_title:
                windows = gw.getWindowsWithTitle(self.selected_window_title)
                visible_windows = [w for w in windows if w.visible]
                if visible_windows:
                    self.window = visible_windows[0]
                    self.console.log(f"[green]{self.loc.get('saved_window_found', self.window.title)}[/green]")
                    logging.info(self.loc.get('saved_window_found', self.window.title))
            
            if not self.window:
                self.window = self.find_telegram_window()
                if not self.window:
                    self.console.log(f"[red]{self.loc.get('window_not_found_exit')}[/red]")
                    logging.error(self.loc.get('window_not_found_exit'))
                    self.is_running = False
                    return

            if not self.bring_window_to_foreground():
                self.console.log(f"[yellow]{self.loc.get('foreground_continue')}[/yellow]")

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
                        old_window_title = self.window.title if self.window else None
                        self.show_settings_panel()
                        if self.window_changed or (self.window and old_window_title != self.window.title) or (old_window_title and not self.window):
                            self.console.log(f"[blue]{self.loc.get('recreating_capture_generator')}[/blue]")
                            capture_gen = self.capture_telegram_window()
                            self.window_changed = False
                        continue

                    if self.pause_signal:
                        time.sleep(0.1)
                        continue

                    start_time = time.time()

                    try:
                        image, bbox = next(capture_gen)
                    except StopIteration:
                        self.console.log(f"[yellow]{self.loc.get('capture_stopped')}[/yellow]")
                        capture_gen = self.capture_telegram_window()
                        time.sleep(0.5)
                        continue
                    except Exception as e:
                        self.console.log(f"[red]{self.loc.get('frame_error', e)}[/red]")
                        time.sleep(0.5)
                        continue

                    if frame_count % self.FRAME_SKIP == 0:
                        try:
                            preprocessed_frame = self.preprocess_image(image)
                            with self.model_lock:
                                with torch.no_grad():
                                    prediction = self.model(preprocessed_frame)
                        except Exception as e:
                            self.console.log(f"[red]{self.loc.get('frame_processing_error', e)}[/red]")
                            logging.error(self.loc.get('frame_processing_error', e))
                            prediction = []
                    else:
                        prediction = []

                    if self.auto_play and (ocr_thread is None or not ocr_thread.is_alive()):
                        ocr_thread = threading.Thread(
                            target=self.detect_play_button,
                            args=(image, bbox, results_queue),
                            daemon=True,
                        )
                        ocr_thread.start()

                    try:
                        play_button_coords = results_queue.get_nowait()
                        if play_button_coords:
                            x, y = play_button_coords
                            absolute_x = x + bbox["left"]
                            absolute_y = y + bbox["top"]

                            self.console.log(f"[green]{self.loc.get('button_click', absolute_x, absolute_y)}[/green]")
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
                                            self.debug_image,
                                            label,
                                            (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.9,
                                            color,
                                            2,
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
                                self.console.log(f"[red]{self.loc.get('model_results_error', e)}[/red]")
                                logging.error(self.loc.get('model_results_error', e))

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
                            self.loc.get("last_click", x, y, count, sum(self.click_counters.values())),
                            title=self.loc.get("click_info_title"),
                            border_style="yellow",
                        )
                    else:
                        click_info_panel = Panel(
                            self.loc.get("no_clicks"),
                            title=self.loc.get("click_info_title"),
                            border_style="yellow",
                        )

                    system_info_panel = Panel(
                        self.loc.get(
                            "system_info_content",
                            avg_fps,
                            self.fps_lock,
                            system_info.get('CPU', 'N/A'),
                            system_info.get('RAM', 'N/A'),
                            system_info.get('GPU', 'N/A'),
                            system_info.get('Всего кликов', '0'),
                            system_info.get('Скорость', '0'),
                            system_info.get('Ошибок', '0'),
                        ),
                        title=self.loc.get("system_info_title"),
                        border_style="green",
                    )

                    status = self.loc.get("paused") if self.pause_signal else self.loc.get("working")
                    auto_play_status = self.loc.get("on") if self.auto_play else self.loc.get("off")
                    sound_status = self.loc.get("on") if self.enable_sound else self.loc.get("off")
                    debug_status = self.loc.get("on") if self.show_debug_window else self.loc.get("off")

                    hotkeys_panel = Panel(
                        f"{self.loc.get('status', status)}\n"
                        f"{self.loc.get('window_short', self.window.title[:15] if self.window else self.loc.get('no_window'))}\n"
                        f"{self.loc.get('autoplay_status', auto_play_status)}\n"
                        f"{self.loc.get('sound_status', sound_status)}\n"
                        f"{self.loc.get('debug_status', debug_status)}\n"
                        f"{self.loc.get('hotkeys')}",
                        title=self.loc.get("controls_title", self.VERSION),
                        border_style="magenta",
                    )

                    layout = Layout()
                    layout.split_row(
                        Layout(name="left"),
                        Layout(self.messages_panel, name="right", ratio=2),
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
            self.console.log(f"[yellow]{self.loc.get('interrupted')}[/yellow]")
            logging.info(self.loc.get('interrupted'))
        except Exception as e:
            self.console.log(f"[red]{self.loc.get('critical_error', e)}[/red]")
            logging.exception(self.loc.get('critical_error', e))

            error_type = type(e).__name__
            error_trace = traceback.format_exc()

            self.console.log(f"[red]{self.loc.get('error_type', error_type)}[/red]")
            self.console.log(f"[red]{self.loc.get('error_trace')}[/red]")
            for line in error_trace.split('\n'):
                if line.strip():
                    self.console.log(f"[dim]{line}[/dim]")

            if "CUDA" in str(e):
                self.console.log(f"[yellow]{self.loc.get('gpu_problem')}[/yellow]")
                self.use_cpu = True
                self.save_settings()
                self.run()
            elif "Permission" in str(e) or "доступ" in str(e).lower():
                self.console.log(f"[yellow]{self.loc.get('permission_problem')}[/yellow]")

        finally:
            self.is_running = False
            self.console.log(f"[blue]{self.loc.get('program_ending')}[/blue]")
            logging.info(self.loc.get('program_terminated'))

            try:
                with open("stats.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "clicks": sum(self.click_counters.values()),
                            "errors": self.error_count,
                            "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "uptime_seconds": time.time() - self.start_time,
                            "version": self.VERSION,
                            "language": self.language,
                        },
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )
                logging.info(self.loc.get('stats_saved'))
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
    loc = Localization()

    print(loc.get("app_title", BlumClicker.VERSION))
    print(loc.get("divider_line"))

    if not check_dependencies():
        sys.exit(1)

    print(loc.get("launching_bot"))
    print(loc.get("exit_info"))
    bot = BlumClicker()
    bot.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        loc = Localization()
        print(loc.get('critical_app_error', e))
        print(loc.get('check_log'))
        logging.critical(loc.get('unhandled_exception', e), exc_info=True)
        traceback.print_exc()
        sys.exit(1)
