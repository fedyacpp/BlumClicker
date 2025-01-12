# BlumClicker

**Latest Update:** December 29, 2024  
No new model weight updates will be released at this time.

BlumClicker is an automation tool for Blum's Drop Game on Telegram, powered by YOLOv11 for near-perfect snowflake detection.  
[Watch the demo.](https://photos.app.goo.gl/caVfEjbUsoawek9J8)  
*(Русская версия README [здесь](https://github.com/fedyacpp/BlumClicker/blob/main/README_ru.md)).*

## Table of Contents

- [Features](#features)  
- [Demo](#demo)  
- [System Requirements](#system-requirements)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Settings](#settings)  
- [Troubleshooting](#troubleshooting)  
- [Roadmap](#roadmap)  
- [Contributing](#contributing)  
- [Contact](#contact)

## Features

- **Automatic Window Detection** – Locates the Telegram game window with no extra setup.  
- **YOLOv11 Object Recognition** – Offers high accuracy in detecting snowflakes.  
- **One-Click Automation** – Instantly clicks on detected targets.  
- **Configurable Settings** – Delays, debug mode, FPS limiter, and more.  
- **Auto Replay** – Automatically restarts when the current game ends.

## Demo

[Video Demo](https://photos.app.goo.gl/caVfEjbUsoawek9J8)

## System Requirements

- **OS:** Windows  
- **Python:** 3.10+ (tested on 3.12.X) 
- **GPU:** NVIDIA RTX (or use CPU with lower performance)  
- **CUDA Toolkit:** Required for GPU acceleration (tested with CUDA 12.5 and 11.8)  
- **Telegram Desktop:** Mandatory for gameplay automation

## Installation

1. **(GPU) Install CUDA Toolkit**  
   - Download from [NVIDIA](https://developer.nvidia.com/cuda-downloads) and follow the instructions.
2. **Clone the Repository**  
   ```bash
   git clone https://github.com/fedyacpp/BlumClicker.git
   cd BlumClicker
   ```
3. **Install Python Dependencies**  
   - **GPU Users:**  
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install -r requirements.txt
     ```
   - **CPU Users:**  
     ```bash
     pip install torch torchvision torchaudio
     pip install -r requirements.txt
     ```
4. **Download `best.pt`**  
   - [Link](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing) and place it in the BlumClicker directory.
5. **Verify CUDA (GPU Only)**  
   ```bash
   python isCudaAvailable.py
   ```
6. **Run**  
   ```bash
   python main.py
   ```

## Usage

1. Open Blum's Drop Game in Telegram Desktop.  
2. Run `python main.py`.  
3. The script detects the game window and automates clicks.  
4. Press **CTRL+Q** twice to stop.

## Settings

- **Delays:** Customize click intervals.  
- **Debug Mode:** Displays model predictions.  
- **FPS Limiter:** Reduces CPU/GPU load.  

## Troubleshooting

- **CUDA Not Detected:** Update drivers and check your GPU compatibility.  
- **Dependency Errors:** Ensure you installed/updated `ultralytics` and the correct `torch` version.  
- **No Model File:** Confirm `best.pt` is in the BlumClicker directory.

## Roadmap

- [x] Additional info in README  
- [x] Improved terminal output  
- [x] Enhanced user-friendliness  
- [x] Higher model accuracy  
- [x] Custom settings  
- [x] Larger dataset  
- [x] Auto-replay  
- [x] Debug mode  
- [x] FPS limiter  
- [ ] Click probability (currently not feasible)  

## Contributing

Contributions are welcome. Fork the repo and create a pull request.

## Contact

- **Discord:** [fedyacpp](https://discord.com/users/fedyacpp)  
- **Telegram:** [@fedyacpp](https://t.me/fedyacpp)
