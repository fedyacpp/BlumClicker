# BlumClicker

**Version:** 2.0.0  
**Last Updated:** February 27, 2025

BlumClicker is an automation tool for Blum's Drop Game on Telegram, powered by YOLOv11 computer vision technology for precise snowflake detection and interaction. The tool features a graphical interface, extensive customization options, and intelligent gameplay automation.

*Русская версия [здесь](https://github.com/fedyacpp/BlumClicker/blob/main/README_ru.md).*

[Watch Demo Video](https://photos.app.goo.gl/caVfEjbUsoawek9J8)

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Hotkeys](#hotkeys)
- [Troubleshooting](#troubleshooting)
- [Advanced Features](#advanced-features)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Contact](#contact)

## Features

- **Window Recognition** - Automatically detects and focuses on the Telegram game window
- **YOLOv11 Object Detection** - State-of-the-art visual recognition for snowflake targets
- **OCR Technology** - Detects and clicks the "Play" button for automatic game restart
- **Real-time Statistics** - Monitor performance metrics, click counts, and system resource usage
- **Rich GUI Interface** - Visualize bot activity with detailed panels and real-time updates
- **Configurable Parameters** - Customize delays, FPS limits, detection thresholds and more
- **Debug Visualization** - Optional visual feedback showing detection boundaries and confidence scores
- **Hotkey Controls** - Convenient keyboard shortcuts for all major functions

## System Requirements

### Minimum Advisable Requirements
- **Operating System:** Windows 10 (64-bit)
- **CPU:** Intel Core i5 or AMD Ryzen 5 equivalent
- **RAM:** 8GB
- **GPU:** NVIDIA GPU with CUDA support (tested with CUDA 11.8 and 12.5)
- **Python:** Version 3.12.x
- **Storage:** 2GB free space
- **Software:** Telegram Desktop application

### Recommended Advisable Configuration
- **Operating System:** Windows 10/11 (64-bit)
- **CPU:** Intel Core i7/i9+ or AMD Ryzen 7/9+
- **RAM:** 16GB+
- **GPU:** NVIDIA GPU with CUDA support (tested with CUDA 11.8 and 12.5)
- **Python:** Version 3.12.x
- **Storage:** 5GB+ storage

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/fedyacpp/BlumClicker.git
   cd BlumClicker
   ```

2. **Install Required Python Packages**
   
   For GPU users (recommended for best performance):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
   
   For CPU users:
   ```bash
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   ```

3. **Download Model File**
   
   Download the pre-trained YOLO model file:
   - [best.pt](https://drive.google.com/file/d/1lUTl4GulseoWs_vhPnYp0qkIYaumKMNg/view?usp=sharing)
   - Place it in the BlumClicker root directory

4. **Verify CUDA Availability** (GPU users only)
   ```bash
   python isCudaAvailable.py
   ```

5. **Launch BlumClicker**
   ```bash
   python main.py
   ```

## Usage

1. **Start the Application**
   ```bash
   python main.py
   ```

2. **Window Detection**
   - The bot will automatically detect your Telegram window

3. **Controls**
   - Use the hotkeys to control the bot (see [Hotkeys](#hotkeys) section)
   - The main interface displays real-time statistics and status information
   - Access settings via CTRL+W to customize behavior

4. **Operation Modes**
   - Auto-Play Mode: Automatically restarts the game when it ends
   - Manual Mode: Requires manual restart of games
   - Debug Mode: Shows visual detection feedback

## Configuration

BlumClicker offers extensive configuration options through its settings panel (CTRL+W):

### Timing Settings
- **Delay Between Clicks** - Time to wait between consecutive clicks (seconds)
- **Delay Before Click** - Time to wait before performing a click (seconds)
- **FPS Limit** - Maximum frames processed per second
- **Retry Count** - Number of retry attempts for operations

### Feature Options
- **Auto-Play** - Automatically start new games
- **Debug Window** - Show visual detection feedback
- **Click All Bombs** - Click on all detected objects, not just targets
- **CPU Mode** - Force CPU-only processing (disable GPU)
- **Sound Effects** - Enable/disable sound feedback for clicks

### Model Settings
- **Model Path** - Location of the YOLO model file
- **Model Reload** - Ability to hot-swap models during runtime

## Hotkeys

| Hotkey | Function |
|--------|----------|
| CTRL+Q | Exit the application |
| CTRL+X | Pause/Resume bot operation |
| CTRL+W | Open settings panel |
| CTRL+D | Toggle debug visualization |
| CTRL+R | Reload the model |
| CTRL+F | Re-detect Telegram window |
| CTRL+A | Toggle auto-play mode |
| CTRL+S | Toggle sound effects |

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **Window not detected** | Ensure Telegram is running and visible; try CTRL+F to re-detect |
| **CUDA errors** | Verify GPU and CUDA installation; switch to CPU mode if needed |
| **OCR not working** | Ensure EasyOCR dependencies are installed properly |
| **Inaccurate detection** | Ensure the model file is correctly loaded and running on supported GPU |
| **Permission errors** | Run as administrator for proper window access |

### Logs and Diagnostics

- Check `bot_log.txt` for detailed error information
- Enable debug mode (CTRL+D) to visualize detection performance
- Use the settings panel to view system resource usage and last errors

## Advanced Features

### Custom Models
You can use your own trained YOLO models by selecting them in the settings panel:
1. Train a custom YOLOv11 model
2. Place the .pt file in an accessible location
3. Set the model path in settings and reload

### Performance Optimization
- **GPU Acceleration**: Enabled by default when available
- **Memory Management**: Configure with appropriate FPS limits
- **CPU Mode**: Available for systems without compatible GPUs

## FAQ

**Q: Is this against Telegram's terms of service?**  
A: Automation tools may violate Telegram's terms of service. Use at your own discretion.

**Q: Can I use multiple instances for different games?**  
A: Technically yes, but it won't work as supposed right out of the box.

**Q: Does BlumClicker work on MacOS/Linux?**  
A: Currently, it's Windows-only due to the specific window handling mechanisms.

**Q: How accurate is the detection?**  
A: With YOLOv11 and proper setup, detection accuracy is typically over 95%.

**Q: Will my account get banned?**  
A: Use at your own risk. The tool tries to simulate human-like behavior, but no guarantees.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

- **GitHub Issues:** [Report bugs or request features](https://github.com/fedyacpp/BlumClicker/issues)
- **Email:** [fedyacpp@protonmail.com](mailto:fedyacpp@protonmail.com)
- **Telegram:** [@fedyacpp](https://t.me/fedyacpp)

---

**Disclaimer:** This tool is for educational purposes only. I'm not responsible for any misuse or violations of terms of service that may result from using this software.
