# BlumClicker

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

2. **Clone the repository:**
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

## Roadmap

- [x] Improved documentation
- [x] Enhanced terminal output
- [x] User-friendly script interface
- [x] Advanced model training
- [x] Customizable settings (delays, etc.)
- [x] Expanded dataset
- [x] Auto-replay functionality
- [x] Comprehensive documentation
- [ ] Debug window with predictions
- [ ] FPS limiting
- [ ] Click probability adjustment

## Troubleshooting

If you encounter issues:
1. Verify your CUDA installation and library versions
2. Check console and log outputs for error messages

For further assistance, open an issue or contact me on [Discord](https://discord.com/users/fedyacpp) or [Telegram](t.me/fedyacpp).
