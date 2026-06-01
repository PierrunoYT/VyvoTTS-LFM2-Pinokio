# VyvoTTS LFM2

High-quality Text-to-Speech powered by the [VyvoTTS LFM2](https://github.com/Vyvo-Labs/VyvoTTS) architecture, with a Gradio web interface for easy use.

## What It Does

Converts text to natural-sounding speech using LFM2 language models fine-tuned on diverse voice datasets. Supports 16 distinct voice characters, with configurable generation parameters.

### Available Voice Models

| Name | HuggingFace ID |
|---|---|
| Jenny | Vyvo/VyvoTTS-LFM2-Jenny |
| Neuvillette | Vyvo/VyvoTTS-LFM2-Neuvillette |
| Optimus Prime | Vyvo/VyvoTTS-LFM2-Optimus-Prime |
| Multi-Speaker | Vyvo/VyvoTTS-LFM2-Multi-Speaker |
| Itto | Vyvo/VyvoTTS-LFM2-Itto |
| Stephen_Fry | Vyvo/VyvoTTS-LFM2-Stephen_Fry |
| Alhaitham | Vyvo/VyvoTTS-LFM2-Alhaitham |
| Cyno | Vyvo/VyvoTTS-LFM2-Cyno |
| Dehya | Vyvo/VyvoTTS-LFM2-Dehya |
| Elise | Vyvo/VyvoTTS-LFM2-Elise |
| Kaeya | Vyvo/VyvoTTS-LFM2-Kaeya |
| Kaveh | Vyvo/VyvoTTS-LFM2-Kaveh |
| Ningguang | Vyvo/VyvoTTS-LFM2-Ningguang |
| Heizou | Vyvo/VyvoTTS-LFM2-Heizou |
| Thoma | Vyvo/VyvoTTS-LFM2-Thoma |
| Tighnari | Vyvo/VyvoTTS-LFM2-Tighnari |

## How to Use

### Via Pinokio (Recommended)

1. Open the app in Pinokio and click **Install** — this sets up the Python environment and installs all dependencies including PyTorch.
2. Click **Start** — the app launches and the **Open Web UI** button appears automatically.
3. Click **Open Web UI** to open the Gradio interface in your browser.
4. Enter your text, pick a voice model, and click **Generate Speech**.
5. Optionally expand **Advanced Settings** to tune Temperature, Top P, Repetition Penalty, and Maximum Length.

### Advanced Settings

| Parameter | Default | Description |
|---|---|---|
| Temperature | 0.3 | Higher = more expressive but less stable |
| Top P | 0.95 | Nucleus sampling threshold |
| Repetition Penalty | 1.2 | Higher = fewer repeated patterns |
| Maximum Length | 2000 | Max tokens generated (affects audio length) |

---

## API

The app exposes a standard Gradio HTTP API at `http://127.0.0.1:<PORT>` (port is assigned automatically on launch — check the terminal or the **Open Web UI** button URL).

### Python

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")

result = client.predict(
    text="Hello! This is a speech synthesis demo.",
    model_choice="Jenny",
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2,
    max_new_tokens=2000,
    api_name="/generate_speech"
)

# result is a tuple: (sample_rate, numpy_array)
print(result)
```

### JavaScript

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");

const result = await client.predict("/generate_speech", {
  text: "Hello! This is a speech synthesis demo.",
  model_choice: "Jenny",
  temperature: 0.3,
  top_p: 0.95,
  repetition_penalty: 1.2,
  max_new_tokens: 2000,
});

console.log(result.data); // [sample_rate, audio_data]
```

### curl

```bash
curl -X POST http://127.0.0.1:7860/run/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fn_index": 0,
    "data": [
      "Hello! This is a speech synthesis demo.",
      "Jenny",
      0.3,
      0.95,
      1.2,
      2000
    ]
  }'
```

The response contains a `data` array with the audio output as a base64-encoded WAV file or a numpy array depending on the Gradio version.

---

## Links

- [VyvoTTS GitHub](https://github.com/Vyvo-Labs/VyvoTTS)
- [HuggingFace Models](https://huggingface.co/collections/Vyvo/lfm2-tts-689eedae5353ff5b048efd55)
