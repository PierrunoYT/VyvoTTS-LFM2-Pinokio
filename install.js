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
          "python -c \"import gradio, snac; from transformers import Lfm2ForCausalLM\"",
        ]
      }
    },
    {
      method: "fs.write",
      params: {
        path: "app/env/.installed",
        text: "Dependencies installed successfully."
      }
    },
  ]
}
