"""The local Hugging Face backend runs end to end on a tiny random Qwen3-VL (no download)."""
import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


def _tiny_qwen3_vl(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
               "<|image_pad|>", "<|video_pad|>"]
    words = ["{", "}", '"verdict"', ":", '"', "yes", "no", "uncertain"]
    vocab = {w: i for i, w in enumerate(special + words + ["[UNK]"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tmpl = ("{% for m in messages %}<|im_start|> {{ m['role'] }} {% for c in m['content'] %}"
            "{% if c['type']=='image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ c['text'] }} "
            "{% endif %}{% endfor %}<|im_end|> {% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}")
    ft = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>",
                                 pad_token="<|endoftext|>", additional_special_tokens=special)
    ip = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=32 * 32, max_pixels=64 * 64)
    proc = Qwen3VLProcessor(image_processor=ip, tokenizer=ft, video_processor=Qwen3VLVideoProcessor(), chat_template=tmpl)
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[0]),
        image_token_id=vocab["<|image_pad|>"], video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"], vision_end_token_id=vocab["<|vision_end|>"])
    m = Qwen3VLForConditionalGeneration(cfg)
    m.generation_config.eos_token_id = vocab["<|im_end|>"]
    m.generation_config.pad_token_id = vocab["<|endoftext|>"]
    m.save_pretrained(path)
    proc.save_pretrained(path)


def test_transformers_backend_runs_a_vad_request(tmp_path):
    from pathlib import Path

    from edgestream.types import Candidate
    from edgestream.vlm import load_cards
    from edgestream.vlm.backends import TransformersBackend
    from edgestream.vlm.prompt_vad import VADPromptBuilder, load_hypotheses
    try:
        _tiny_qwen3_vl(tmp_path / "m")
    except Exception as e:                     # API drift across transformers versions
        pytest.skip(f"cannot build a tiny Qwen3-VL with transformers {transformers.__version__}: {e}")
    cards = Path(__file__).resolve().parents[1] / "event_cards" / "uav_surveillance"
    b = TransformersBackend(str(tmp_path / "m"), device="cpu")
    pb = VADPromptBuilder(load_cards(cards), load_hypotheses(cards), max_new_tokens=8)
    ctx = {"stats_text": "x", "scene_log": [],
           "tracks": [{"cls_name": "pedestrian", "trajectory": "fast",
                       "crops": [(np.zeros((224, 224, 3), np.uint8), (224, 224), 1.0)]}]}
    req = pb.build(Candidate(1, "running", [1], 0, 3, .8, .8, .5), ctx, np.zeros((360, 640, 3), np.uint8),
                   [(10, 10, 20, 40)])
    v = b(req)                                  # random weights: any verdict, but the call must work
    assert v.verdict in ("yes", "no", "uncertain") and v.prompt_tokens > 100 and v.completion_tokens >= 1
