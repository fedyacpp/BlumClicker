## BlumClicker

**Heads up: I've updated the model weights download link. Re-download the weights file even if you've grabbed it before.**

Achieve ~100% accuracy [watch here](https://photos.app.goo.gl/TYiW38Hc1g3Qqbnu5).

*Русский README [здесь](https://github.com/fedyacpp/BlumClicker/blob/main/README_ru.md).*

BlumClicker automates interactions with Telegram using image-based object detection via YOLOv5. It's designed to spot snowflakes in Blum's Drop Game.

## Requirements

* **NVIDIA GPU (Recommended):** From any other GPUs, this script will work only with an NVIDIA RTX series. YOLOv5 relies heavily on CUDA cores.
* **CPU (Alternative):** You can run the script on a CPU if you don't have an NVIDIA GPU, but expect it to be much slower and less accurate.

**Note:** You need the desktop version of Telegram for this script to work.

## TODO

- [x] Add more info in readme
- [x] Show more info in terminal
- [x] Make script more user-friendly
- [x] Train model for even better accuracy
- [x] Add settings (e.g., custom delay between/before clicks)
- [ ] Collect even more photos in dataset
- [ ] Add click probability
- [ ] Add even MORE info in readme

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

3. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   If you're using a CPU (much slower image processing):
   ```bash
   pip install torch torchvision torchaudio
   ```

4. **Install the remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the model weights:**
   - Get the weights from this [link](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing).
   - Place the `best.pt` file in the same directory as the script, or specify its path when running the script.

### Check CUDA Installation (for NVIDIA GPU users)

To verify CUDA installation, run the `isCudaAvailable.py` script in the repo:

```bash
python isCudaAvailable.py
```

If everything's set up right, you'll see your CUDA and GPU info.

### Running the Script

Ensure the Blum window is open in your Telegram desktop app before running the script.

To start the script, just run:
```bash
python main.py
```

## Exiting the Script

To exit, press `CTRL+Q` twice.

## Features

- **Telegram Window Search:** Finds an open Telegram window on your desktop.
- **Object Detection:** Uses YOLOv5 to detect objects in the Telegram window.
- **Automated Clicking:** Clicks on detected objects.

## Troubleshooting

If you run into issues, double-check your CUDA installation and library versions. Pay attention to error messages in the console and logs.

Need help? Open an issue or message me on [Discord](https://discord.com/users/fedyacpp)/[Telegram](t.me/fedyacpp).
