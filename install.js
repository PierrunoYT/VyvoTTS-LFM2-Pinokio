module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      when: "{{exists('app/env/.installed')}}",
      method: "fs.rm",
      params: {
        path: "app/env/.installed"
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip check",
          // Pinokio chains these with && and ignores exit codes, so only the
          // final command may write the marker that enables Start.
          "python -c \"import gradio, snac; from transformers import Lfm2ForCausalLM; from pathlib import Path; Path('env/.installed').write_text('Dependencies installed successfully.')\"",
        ]
      }
    },
  ]
}
