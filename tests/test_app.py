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

    def parse(self, tokens):
        tensor = MagicMock()
        tensor.__getitem__.return_value.tolist.return_value = tokens
        return self.app.parse_output(tensor)

    def frame(self):
        return [self.app.AUDIO_TOKENS_START + i * 4096 + i for i in range(7)]

    def test_parser_stops_at_end_of_speech(self):
        codes = self.parse([123, self.app.START_OF_SPEECH] + self.frame() +
                           [self.app.END_OF_SPEECH, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(codes, [i * 4096 + i for i in range(7)])

    def test_parser_trims_partial_final_frame(self):
        codes = self.parse([self.app.START_OF_SPEECH] + self.frame() + self.frame()[:3])
        self.assertEqual(len(codes), 7)

    def test_parser_rejects_missing_empty_and_invalid_speech(self):
        cases = [self.frame(), [self.app.START_OF_SPEECH, self.app.END_OF_SPEECH],
                 [self.app.START_OF_SPEECH] + self.frame()[:6]]
        for position in range(7):
            for delta in [-1, 4096]:
                frame = self.frame()
                frame[position] = self.app.AUDIO_TOKENS_START + position * 4096 + delta
                cases.append([self.app.START_OF_SPEECH] + frame)
        for tokens in cases:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.parse(tokens)

    def test_decoder_redistributes_layers_and_disables_gradients(self):
        codec = MagicMock()
        codec.parameters.return_value = iter([MagicMock(device="cpu")])
        self.app.redistribute_codes([i * 4096 + i for i in range(7)], codec)
        arrays = [call.args[0] for call in self.torch.tensor.call_args_list]
        self.assertEqual(arrays, [[0], [1, 4], [2, 3, 5, 6]])
        self.torch.inference_mode.return_value.__enter__.assert_called_once()
        codec.decode.assert_called_once()

    def test_decoder_rejects_invalid_codes_before_decoding(self):
        for codes in [[], [0] * 6, [-1] * 7, [0] * 7]:
            codec = MagicMock()
            with self.assertRaises(ValueError):
                self.app.redistribute_codes(codes, codec)
            codec.decode.assert_not_called()

    def test_api_rejects_invalid_parameters_before_loading(self):
        defaults = ["Hello", "Jenny", 0.3, 0.95, 1.2, 2000]
        for index, value in [(0, None), (0, "  "), (1, "unknown"),
                             (2, float("nan")), (2, 0), (3, 2),
                             (4, float("inf")), (5, 2001), (5, 100.5), (5, True)]:
            args = defaults.copy()
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(self.gradio.Error):
                self.app.generate_speech(*args)
        self.app.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_generation_reports_errors_and_runs_cleanup(self):
        with patch.object(self.app, "load_model_if_needed", side_effect=RuntimeError("offline")), \
             patch.object(self.app.gc, "collect") as collect, \
             patch.object(self.app.logging, "exception"):
            with self.assertRaises(self.gradio.Error):
                self.app.generate_speech("Hello", "Jenny", 0.3, 0.95, 1.2, 2000)
            collect.assert_called_once()

    def test_generation_parses_only_continuation_and_normalizes_length(self):
        model, tokenizer = MagicMock(), MagicMock()
        ids = MagicMock()
        ids.shape = (1, 12)
        with patch.object(self.app, "load_model_if_needed", return_value=(model, tokenizer)), \
             patch.object(self.app, "process_prompt", return_value=(ids, MagicMock())), \
             patch.object(self.app, "parse_output", return_value=[0]) as parse, \
             patch.object(self.app, "load_snac_if_needed"), \
             patch.object(self.app, "redistribute_codes", return_value="audio"):
            result = self.app.generate_speech("Hello", "Jenny", 0.3, 0.95, 1.2, 2000.0)
        self.assertEqual(result, (24000, "audio"))
        self.assertIs(type(model.generate.call_args.kwargs["max_new_tokens"]), int)
        model.generate.return_value.__getitem__.assert_called_once_with((slice(None), slice(12, None)))
        parse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
