"""Offline regression tests; ML downloads and the Gradio UI are mocked."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch


class AppTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = False
        self.gradio = MagicMock()
        self.gradio.Error = type("GradioError", (Exception,), {})
        modules = {"torch": self.torch, "gradio": self.gradio,
                   "snac": MagicMock(), "transformers": MagicMock()}
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            "tts_app", Path(__file__).resolve().parents[1] / "app" / "app.py")
        self.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.app)

    def test_import_does_not_download_models(self):
        self.app.SNAC.from_pretrained.assert_not_called()
        self.app.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_failed_switch_can_retry_old_and_new_voice(self):
        self.app.load_model_if_needed("Jenny")
        factory = self.app.AutoTokenizer.from_pretrained
        factory.side_effect = RuntimeError("download failed")
        with self.assertRaises(RuntimeError):
            self.app.load_model_if_needed("Cyno")
        self.assertIsNone(self.app.current_model)
        self.assertIsNone(self.app.current_model_choice)
        factory.side_effect = None
        self.app.load_model_if_needed("Jenny")
        self.app.load_model_if_needed("Cyno")
        self.assertEqual(self.app.current_model_choice, "Cyno")

    def test_device_transfer_failure_does_not_publish_cache(self):
        self.app.AutoModelForCausalLM.from_pretrained.return_value.to.side_effect = RuntimeError("OOM")
        with self.assertRaises(RuntimeError):
            self.app.load_model_if_needed("Jenny")
        self.assertIsNone(self.app.current_model)
        self.assertIsNone(self.app.current_tokenizer)

    def test_unknown_voice_preserves_loaded_voice(self):
        model, tokenizer = self.app.load_model_if_needed("Jenny")
        with self.assertRaises(ValueError):
            self.app.load_model_if_needed("missing")
        self.assertIs(self.app.current_model, model)
        self.assertIs(self.app.current_tokenizer, tokenizer)

    def test_cached_voice_is_reused(self):
        self.app.load_model_if_needed("Jenny")
        self.app.load_model_if_needed("Jenny")
        self.app.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_dtype_matches_device_capabilities(self):
        for device, bf16, expected in [("cpu", False, self.torch.float32),
                                       ("cuda", False, self.torch.float16),
                                       ("cuda", True, self.torch.bfloat16)]:
            with self.subTest(device=device, bf16=bf16):
                self.app.device = device
                self.app.current_model_choice = None
                self.torch.cuda.is_bf16_supported.return_value = bf16
                self.app.load_model_if_needed("Jenny")
                self.assertIs(self.app.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"], expected)

    def test_snac_load_failure_is_retryable(self):
        self.app.SNAC.from_pretrained.side_effect = RuntimeError("offline")
        with self.assertRaises(RuntimeError):
            self.app.load_snac_if_needed()
        self.assertIsNone(self.app.snac_model)
        self.app.SNAC.from_pretrained.side_effect = None
        first = self.app.load_snac_if_needed()
        self.assertIs(first, self.app.load_snac_if_needed())


if __name__ == "__main__":
    unittest.main()
