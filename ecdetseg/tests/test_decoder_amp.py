import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.edgecrafter.decoder import TransformerDecoderLayer  # noqa: E402


class RecordingAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.autocast_enabled = None
        self.input_dtypes = None

    def forward(self, query, key, value, attn_mask=None):
        self.autocast_enabled = torch.is_autocast_enabled(query.device.type)
        self.input_dtypes = (query.dtype, key.dtype, value.dtype)
        return query, None


class DecoderAmpTest(unittest.TestCase):
    def test_self_attention_runs_in_float32_inside_autocast(self):
        layer = TransformerDecoderLayer(
            d_model=8,
            n_head=2,
            dim_feedforward=16,
            n_levels=1,
            n_points=1,
        )
        attention = RecordingAttention()
        layer.self_attn = attention
        target = torch.ones((1, 2, 8), dtype=torch.bfloat16)
        query_pos_embed = torch.ones_like(target)

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            result = layer._self_attention_forward(target, query_pos_embed, None)

        self.assertFalse(attention.autocast_enabled)
        self.assertEqual(attention.input_dtypes, (torch.float32,) * 3)
        self.assertEqual(result.dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
