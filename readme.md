# VyvoTTS LFM2 - Pinokio

A Pinokio-compatible Text-to-Speech application powered by VyvoTTS LFM2 model from Hugging Face Spaces, providing high-quality speech synthesis with an easy-to-use web interface.

## Features

- 🎙️ High-quality, natural-sounding speech synthesis powered by VyvoTTS LFM2
- 🌐 Web-based Gradio interface
- 🔧 Easy installation through Pinokio
- 💾 Automatic audio file saving
- 🚀 One-click setup and launch
- 📱 User-friendly interface

## Prerequisites

- [Pinokio](https://pinokio.computer/) installed on your system
- Windows, macOS, or Linux
- Internet connection for downloading models

## Installation

### Method 1: Through Pinokio (Recommended)

1. **Install Pinokio** from [pinokio.computer](https://pinokio.computer/)
2. **Open Pinokio** and navigate to the "Discover" section
3. **Search for "VyvoTTS LFM2"** or use this repository URL
4. **Click "Install"** and wait for the automatic setup
5. **Click "Start"** to launch the application
6. **Click "Open Web UI"** when the server is ready

### Method 2: Manual Installation

1. **Clone this repository:**
```bash
git clone https://github.com/PierrunoYT/VyvoTTS-LFM2-Pinokio.git
cd VyvoTTS-LFM2-Pinokio
```

2. **Install through Pinokio:**
   - Open Pinokio
   - Click "Install" from the local folder
   - Follow the installation process

## Usage

1. **Launch the application** through Pinokio
2. **Open the Web UI** when prompted
3. **Enter your text** in the input field
4. **Adjust settings** as needed (voice, speed, etc.)
5. **Click "Generate"** to create speech
6. **Download or play** the generated audio

## Model Information

### VyvoTTS LFM2
- **Source:** Hugging Face Space `Vyvo/VyvoTTS-LFM2`
- **Type:** Advanced Text-to-Speech model
- **Interface:** Gradio web application
- **Features:** High-quality speech synthesis
- **License:** Check the original Hugging Face Space for licensing details

## File Structure

```
VyvoTTS-LFM2-Pinokio/
├── install.js                # Pinokio installation script
├── start.js                  # Pinokio startup script
├── pinokio.js               # Pinokio configuration
├── README.md                # This file
├── app/                     # VyvoTTS LFM2 application (auto-downloaded)
│   ├── app.py              # Main application file
│   ├── requirements.txt    # Python dependencies
│   └── ...                 # Other model files
└── env/                     # Python virtual environment (auto-created)
```

## Troubleshooting

### Installation Issues
- **Slow download:** The VyvoTTS LFM2 model files may be large. Please be patient during first-time setup.
- **Network errors:** Ensure stable internet connection for downloading from Hugging Face.
- **Permission errors:** Make sure Pinokio has proper permissions to create files and folders.

### Runtime Issues
- **Application won't start:** Check that all dependencies were installed correctly.
- **Web UI not accessible:** Wait a few moments for the server to fully start up.
- **Audio generation fails:** Check system resources and try again.

## Support

For issues related to:
- **Pinokio setup:** Check the [Pinokio documentation](https://docs.pinokio.computer/)
- **VyvoTTS LFM2 model:** Visit the [original Hugging Face Space](https://huggingface.co/spaces/Vyvo/VyvoTTS-LFM2)
- **This implementation:** Open an issue in this repository

## Credits

- **VyvoTTS LFM2:** Original model and application by [Vyvo](https://huggingface.co/spaces/Vyvo/VyvoTTS-LFM2)
- **Pinokio Integration:** Adapted for Pinokio by PierrunoYT
- **Pinokio:** [Pinokio Computer](https://pinokio.computer/) for the amazing platform

## License

This Pinokio integration is provided as-is. Please refer to the original VyvoTTS LFM2 Hugging Face Space for model licensing information.


