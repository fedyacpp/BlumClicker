## BlumClicker

**Important: The model weights download link has been updated. Please re-download the weights file even if you have downloaded it before.**

~100% accuracy [video](https://photos.app.goo.gl/YBA8ETyRXX5Evym99)

*RU README [here](https://github.com/fedyacpp/BlumClicker/blob/main/README_ru.md).*

BlumClicker is a tool that automates interacting with Telegram using image-based object detection powered by YOLOv5. It’s specifically trained to spot snowflakes in Blum's Drop Game.

## Requirements

* **NVIDIA GPU (Recommended):** For optimal performance, an NVIDIA graphics card from the RTX 20 series or newer is highly recommended. This is due to the reliance on CUDA cores for YOLOv5's object detection. 
* **CPU (Alternative):**  If you don't have a compatible NVIDIA GPU, you can still run the script using your CPU. However, be aware that the frame rate will be significantly lower, potentially leading to inaccurate clicks as the snowflakes might move before the script reacts.

## TODO

- [x] Add more info in readme
- [x] Show more info in terminal
- [x] Make script more user-friendly
- [x] Train model for even better accuracy
- [ ] Add settings (for custom delay between/before clicks for example)
- [ ] Add even MORE info in readme

## Installation

You'll need Python 3.x and CUDA if you want to use GPU support. This has been tested with Python 3.10 and CUDA 12.5.

### How to Install

1. **Get CUDA Toolkit (for NVIDIA GPUs):**

   To use GPU support, you need to install the CUDA Toolkit. Here’s how:

   - Head over to the [CUDA Toolkit Download Page](https://developer.nvidia.com/cuda-downloads).
   - Pick your operating system and grab the right installer for your setup.
   - Download and run the installer, then just follow the steps to get it set up.

2. **Clone the Repo:**
   ```bash
   git clone https://github.com/fedyacpp/BlumClicker
   cd BlumClicker
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   If you don’t have an NVIDIA GPU and need to run on CPU (which will be slower):
   ```bash
   pip install torch torchvision torchaudio
   ```

4. **Install the rest of the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the model weights:**
   - Download the weights from this [link](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing).
   - Place the downloaded weights file (`best.pt`) in the same directory as the script, or specify its path when you run the script.

### Check CUDA Installation (for NVIDIA GPU users)

To make sure CUDA is installed properly, run the `isCudaAvailable.py` script included in the repo:

```bash
python isCudaAvailable.py
```

This will check if your CUDA drivers are good to go. If everything's set up right, you’ll see a message with your CUDA and GPU info.

### Running the Script

Make sure you have the Blum window open in your Telegram app before running the script. Otherwise, it won't work.

To start the script, just run:
```bash
python main.py
```

## Exiting the Script

To exit the script gracefully, press `CTRL+Q` twice.

## Features

- **Telegram Window Search:** Automatically finds an open Telegram window on your computer.
- **Object Detection:** Uses YOLOv5 to spot objects in the Telegram window.
- **Automated Clicking:** Clicks on the detected objects.

## Troubleshooting

If you hit any snags, double-check that CUDA is installed correctly and that your library versions match your CUDA version. Pay attention to error messages in the console and logs.

If you’re stuck or have questions, open an issue or hit me up on [Discord](https://discord.com/users/fedyacpp). 
