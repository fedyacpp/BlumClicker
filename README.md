# BlumClicker

## Dev info: as for July 4 blum got collab with some people and changed everything including freeze, point and bomb sprites, so tomorrow I'll retrain model and upload new weights

## Latest Update: June 20, 2024 - YOLOv8 Upgrade
Please redownload the model weights!

BlumClicker is an automation tool designed to interact with Telegram's Blum's Drop Game using YOLOv8-based image detection. It can achieve near-perfect accuracy in identifying and clicking on snowflakes. [Watch it in action](https://photos.app.goo.gl/TYiW38Hc1g3Qqbnu5).

*Русская версия README доступна [здесь](https://github.com/fedyacpp/BlumClicker/blob/main/README_ru.md).*

## System Requirements

- **Recommended:** NVIDIA GPU (RTX series) for optimal performance
- **Alternative:** CPU (slower and less accurate)

**Note:** This script requires the desktop version of Telegram.

## Quick Start Guide

1. **Install CUDA Toolkit** (for NVIDIA GPUs):
   - Download from [NVIDIA's CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation instructions for your OS

**Note:** You need the desktop version of Telegram for this script to work.

## TODO

- [x] Add more info in readme
- [x] Show more info in terminal
- [x] Make script more user-friendly
- [x] Train model for even better accuracy
- [x] Add settings (e.g., custom delay between/before clicks)
- [x] Collect even more photos in dataset
- [x] Auto play again
- [x] Add Debug window with predicts
- [x] Add fps lock
- [ ] Add click probability
- [x] Add even MORE info in readme

## Installation

You'll need Python 3.x and CUDA for GPU support. Tested with Python 3.10 and CUDA 12.5.

### Steps to Install

1. **Get CUDA Toolkit (for NVIDIA GPUs):**
   - Visit the [CUDA Toolkit Download Page](https://developer.nvidia.com/cuda-downloads).
   - Choose your OS and get the right installer.
   - Download, run the installer, and follow the setup steps.

2. **Clone the Repo:**
   ```bash
   git clone https://github.com/fedyacpp/BlumClicker
   cd BlumClicker
   ```

3. **Install dependencies:**
   - For GPU users:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install -r requirements.txt
     ```
   - For CPU users:
     ```bash
     pip install torch torchvision torchaudio
     pip install -r requirements.txt
     ```

4. **Download model weights:**
   - Get `best.pt` from [this link](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing)
   - Place it in the script directory

5. **Verify CUDA installation** (GPU users):
   ```bash
   python isCudaAvailable.py
   ```

6. **Run the script:**
   ```bash
   python main.py
   ```

## Usage

- Ensure the Blum game window is open in Telegram desktop before running the script
- To exit, press `CTRL+Q` twice

## Features

- Automatic Telegram window detection
- YOLOv8-powered object recognition
- Automated clicking on detected snowflakes

## Troubleshooting

If you encounter issues:
1. Verify your CUDA installation and library versions
2. Check console and log outputs for error messages

For further assistance, open an issue or contact me on [Discord](https://discord.com/users/fedyacpp) or [Telegram](t.me/fedyacpp).
