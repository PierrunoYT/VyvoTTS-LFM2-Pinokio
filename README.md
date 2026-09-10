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

The first generation downloads the selected voice and SNAC audio decoder from Hugging Face. Later runs reuse the cache. Generation uses CUDA/ROCm when PyTorch reports a compatible device, and otherwise uses CPU (including Windows AMD and macOS). Requests run one at a time because voices share model memory.

**Update** pulls launcher/app changes and reruns dependency installation. **Reset** removes the Python environment; it retains the shared Hugging Face model cache. Existing installations from before the installation-completion check must run **Install** once to enable **Start** again. A failed installation keeps **Install** available for retry.

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

# result is the local path of the downloaded audio file.
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

console.log(result.data[0]); // Audio file metadata, including its URL.
```

### curl

```bash
curl -X POST http://127.0.0.1:7860/gradio_api/call/generate_speech \
  -H "Content-Type: application/json" \
  -d '{
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

The POST returns an `event_id`. Replace `<EVENT_ID>` below with that value, then fetch the result stream:

```bash
curl -N http://127.0.0.1:7860/gradio_api/call/generate_speech/<EVENT_ID>
```

The `complete` event contains an array with audio file metadata, including a downloadable `url`. Failures produce an error event. The HTTP API serializes audio as a file; the Python callback's `(sample_rate, numpy_array)` tuple is internal to the app. See the [Gradio cURL guide](https://www.gradio.app/guides/querying-gradio-apps-with-curl) for the two-request flow.

Replace `7860` in all examples with the port printed by your running app. Install `gradio_client` for Python or `@gradio/client` for JavaScript before using those examples.

## Development checks

```bash
python -m unittest discover -s tests -v
node --test tests/launchers.test.js
```

These offline tests mock ML downloads and Gradio for Python logic tests, and simulate Pinokio menu/platform states. They do not verify model quality or GPU inference. Installation also checks dependency consistency and imports the ML/UI libraries before enabling Start.

With the app dependencies installed, run `python tests/smoke_runtime.py` to check real LFM2 CPU execution, SNAC decoding, and the Gradio client/cURL endpoint flow. This starts a temporary loopback server and uses a tiny random codec and synthetic tokens, without downloading voice weights.

---

## Links

- [VyvoTTS GitHub](https://github.com/Vyvo-Labs/VyvoTTS)
- [HuggingFace Models](https://huggingface.co/collections/Vyvo/lfm2-tts-689eedae5353ff5b048efd55)
