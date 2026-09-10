"""Exercise real libraries and HTTP serialization without downloading weights.

Run after installing app requirements and CPU PyTorch:
    python tests/smoke_runtime.py
The small, randomly initialized codec produces test audio, not usable speech.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import numpy as np
import torch
from gradio_client import Client
from snac import SNAC
from transformers import Lfm2Config, Lfm2ForCausalLM

from app import app


def main():
    assert app.snac_model is None
    assert app.current_model is None
    # Verify the selected Transformers version can run LFM2 on CPU.
    config = Lfm2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, layer_types=["conv", "full_attention"])
    with torch.inference_mode():
        result = Lfm2ForCausalLM(config).eval()(torch.tensor([[1, 2, 3]]))
    assert result.logits.shape == (1, 3, 32)

    codec = SNAC(sampling_rate=24000, encoder_dim=8, encoder_rates=[2, 2],
                 decoder_dim=32, decoder_rates=[2, 2], attn_window_size=None,
                 vq_strides=[4, 2, 1], noise=False).eval()
    frame = [app.AUDIO_TOKENS_START + i * 4096 + i for i in range(7)]
    continuation = torch.tensor([[app.START_OF_SPEECH] + frame * 8 + [app.END_OF_SPEECH]])
    codes = app.parse_output(continuation)
    samples = app.redistribute_codes(codes, codec)
    assert samples.ndim == 1 and samples.size > 0 and np.isfinite(samples).all()

    model = SimpleNamespace(generate=lambda **kw: torch.cat([kw["input_ids"], continuation], dim=1))
    tokenizer = lambda *args, **kw: SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))
    args = ["Hello", "Jenny", 0.3, 0.95, 1.2, 2000]
    with patch.object(app, "device", "cpu"), \
         patch.object(app, "load_model_if_needed", return_value=(model, tokenizer)), \
         patch.object(app, "load_snac_if_needed", return_value=codec):
        try:
            _, url, _ = app.demo.queue().launch(server_name="127.0.0.1", share=False,
                                                ssr_mode=False, prevent_thread_lock=True)
            client = Client(url, verbose=False)
            audio_path = client.predict(*args, api_name="/generate_speech")
            assert Path(audio_path).is_file()
            with httpx.Client(base_url=url, timeout=60) as http:
                response = http.post("/gradio_api/call/generate_speech", json={"data": args})
                response.raise_for_status()
                event_id = response.json()["event_id"]
                events = http.get(f"/gradio_api/call/generate_speech/{event_id}")
                events.raise_for_status()
                assert "event: complete" in events.text, events.text
                assert '"url"' in events.text, events.text
            print("PASS: LFM2 CPU forward, SNAC decoding, Gradio client and HTTP call API")
        finally:
            app.demo.close()


if __name__ == "__main__":
    main()
