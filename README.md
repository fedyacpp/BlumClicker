# BlumClicker

## Latest Update: November 5, 2024

**Important:**

- **Halloween Event Ended:** The Halloween promotion has concluded, and old model weights have been returned. Please redownload the weights file from the [link in step 4](#installation).
- **No Updates for U.S. Elections:** There will be no updates to the weights for the upcoming U.S. elections. Reason: I have personal life, I'm a human just like you.
- **Library Update:** Please run `pip install --upgrade ultralytics` in the script's folder to update the required libraries.

BlumClicker is an automation tool designed to interact with Blum's Drop Game on Telegram using YOLOv11-based image detection. It achieves near-perfect accuracy in identifying and clicking on snowflakes. [Watch the demo](https://photos.app.goo.gl/caVfEjbUsoawek9J8).

*Русская версия README [здесь](https://github.com/fedyacpp/BlumClicker/blob/main/README_ru.md).*

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

- **Automatic Telegram Window Detection:** No manual setup required; the script automatically finds the game window.
- **YOLOv11-Powered Object Recognition:** Utilizes advanced technology for high accuracy.
- **Automated Snowflake Clicking:** Quickly and efficiently clicks on detected objects.
- **Customizable Settings:** Ability to adjust delays, enable debug mode, and more.
- **Auto Replay Feature:** Automatically starts a new game after the current one ends.

## Demo

[Watch BlumClicker in action](https://photos.app.goo.gl/caVfEjbUsoawek9J8).

## System Requirements

- **Operating System:** Windows
- **Python:** Version 3.10 or higher
- **Hardware:**
  - **Recommended:** NVIDIA GPU (RTX series) for best performance
  - **Alternative:** CPU (may run slower and less accurately)
- **Additional:**
  - **CUDA Toolkit:** For GPU users (tested with CUDA 12.5)
  - **Telegram Desktop:** The script works only with the desktop version of Telegram

## Installation

Follow these steps to install BlumClicker on your device.

### Step 1: Install CUDA Toolkit (GPU Users Only)

If you have an NVIDIA graphics card, it's recommended to install the CUDA Toolkit.

- **Download CUDA Toolkit:**
  - Go to the [official website](https://developer.nvidia.com/cuda-downloads) and select your operating system.
- **Install CUDA Toolkit:**
  - Run the downloaded installer and follow the instructions.

### Step 2: Clone the Repository

```bash
git clone https://github.com/fedyacpp/BlumClicker.git
cd BlumClicker
```

### Step 3: Install Python Dependencies

Make sure to upgrade the `ultralytics` library:

```bash
pip install --upgrade ultralytics
```

#### For GPU Users:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### For CPU Users:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Step 4: Download Model Weights

- **Download `best.pt`:**
  - [Download link](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing)
- **Place the File:**
  - Move `best.pt` into the BlumClicker directory

### Step 5: Verify CUDA Installation (GPU Users Only)

```bash
python isCudaAvailable.py
```

You should see a message confirming CUDA availability.

### Step 6: Run the Script

```bash
python main.py
```

## Usage

1. **Open Blum's Drop Game in Telegram Desktop.**
2. **Run the Script:** Execute `python main.py`.
3. **Start Playing:** The script will automatically detect the game window and begin interaction.
4. **Stop the Script:** Press `CTRL+Q` twice.

## Settings

BlumClicker offers a range of settings for customization:

- **Delay Between Clicks:** Set custom delays.
- **Debug Mode:** Enable to display model predictions.
- **FPS Limiter:** Reduces system load.

## Troubleshooting

- **CUDA Issues:**
  - Ensure your GPU supports CUDA and it's installed correctly.
- **Dependency Errors:**
  - Check the versions of installed libraries.
  - Make sure you have upgraded `ultralytics` by running `pip install --upgrade ultralytics`.
- **Model Weights Update:**
  - Make sure you're using the latest `best.pt` from the [link](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing).

If issues persist, open an issue on GitHub or contact me directly.

## Roadmap

- [x] Expand README with additional information
- [x] Improve terminal output for more information
- [x] Enhance user-friendliness
- [x] Train the model for increased accuracy
- [x] Add customizable settings
- [x] Expand the dataset
- [x] Implement auto-replay
- [x] Introduce debug mode
- [x] Add FPS limiter
- [ ] Implement click probability (not possible because of method of recognition—it's discrete)
- [x] Provide even more information in the README

## Contributing

Want to help improve BlumClicker? Fork the repository and submit a pull request with your suggestions.

## Contact

- **Discord:** [fedyacpp](https://discord.com/users/fedyacpp)
- **Telegram:** [@fedyacpp](https://t.me/fedyacpp)
