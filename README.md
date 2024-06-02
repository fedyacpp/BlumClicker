# BlumClicker

BlumClicker is an automated solution designed to interact with the Telegram interface through image-based object detection powered by YOLOv5, specially trained to detect snowflakes in Blum's Drop Game.

## Installation

The project requires Python 3.x and CUDA for GPU support (if GPU capabilities are utilized). It has been tested with Python 3.10 and CUDA 12.5.

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/fedyacpp/BlumClicker
   cd BlumClicker
   ```

2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For installation without CUDA support (CPU-only):
   ```bash
   pip install torch torchvision torchaudio
   ```

3. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the model weights:
   - Download the weights from this [link](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing).
   - Place the downloaded weights file in the same directory as the script, or specify the path to the weights file when running the script.

### Verifying CUDA Installation

To ensure that CUDA drivers are correctly installed, you can run the `isCudaAvailable.py` script included in the repository. Simply execute the following command:

```bash
python isCudaAvailable.py
```

This script will check the availability and proper functioning of CUDA drivers on your system. If everything is set up correctly, you should see a message displaying both your CUDA's and gpu's info.

### Running the Script

To start the script, run:
```bash
python main.py
```

## Features

- **Telegram Window Search:** The script automatically locates an open Telegram window on your computer.
- **Object Detection:** Utilizes YOLOv5 to detect objects within the Telegram window.
- **Automated Clicking:** Performs clicks on the detected objects.
- **Graceful Exit:** To exit, press `CTRL+Q` twice.

## Troubleshooting

If you encounter any issues, ensure that CUDA is correctly installed and that the library versions used are compatible with your CUDA version. Pay attention to error messages in the console and logs.

For problems or questions, you can either open an issue or contact me on Discord: [https://discord.com/users/fedyacpp](https://discord.com/users/fedyacpp).
