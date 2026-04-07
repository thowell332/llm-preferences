"""
Smoke test: load a tiny HF causal LM and run `residual_stream_at_positions`.

Requires: torch, transformers, network (first run downloads ~few MB from Hugging Face Hub).

From `utility_analysis/experiments/linear_probes`:

  python -m unittest discover -s tests -v

Or:

  cd tests && python -m unittest test_activations -v
"""
from __future__ import annotations

import json
import unittest

# conftest.py path hack runs when using unittest discover from this folder;
# ensure path when running the file directly too.
import sys
from pathlib import Path

_lp = Path(__file__).resolve().parents[1]
_ua = _lp.parents[1]
for _p in (_ua, _lp):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import torch
except ImportError:
    torch = None

# Transformers' tiny random Llama (same family as production runs; supports num_logits_to_keep).
TINY_LLAMA_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestResidualActivations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA_ID)
            if cls.tokenizer.pad_token_id is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
            try:
                cls.model = AutoModelForCausalLM.from_pretrained(
                    TINY_LLAMA_ID,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            except TypeError:
                cls.model = AutoModelForCausalLM.from_pretrained(
                    TINY_LLAMA_ID,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
        except Exception as e:
            raise unittest.SkipTest(f"Could not download/load {TINY_LLAMA_ID}: {e}") from e
        cls.model.eval()

    def test_residual_stream_shapes_and_json(self):
        from lp.activations import residual_stream_at_positions

        n_layers = int(self.model.config.num_hidden_layers)
        layers = [0, n_layers - 1] if n_layers > 1 else [0]

        enc = self.tokenizer(
            "Hello world",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=32,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        gen_ids_json, _none, resid_prompt_last, resid_gen_first = residual_stream_at_positions(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            layers=layers,
            max_new_tokens_for_parsing=2,
        )

        ids_list = json.loads(gen_ids_json)
        self.assertGreaterEqual(len(ids_list), 1)

        hidden = int(self.model.config.hidden_size)
        for lid in layers:
            self.assertIn(lid, resid_prompt_last)
            self.assertIn(lid, resid_gen_first)
            self.assertEqual(resid_prompt_last[lid].shape, (hidden,))
            self.assertEqual(resid_gen_first[lid].shape, (hidden,))


if __name__ == "__main__":
    unittest.main()
