# Tests

A fast, GPU-free suite covering the parts of the release that can be checked without
loading an 8B model: the CLI surface, the `--max_samples` cap, tokenisation and label
masking, the collate contract, and the token projectors.

```bash
uv sync --extra test
uv run pytest tests -q
```

`unsloth` is stubbed in `conftest.py`, so the suite runs on a laptop even though the
training entrypoint imports it at module scope.

## What is deliberately not covered

Model loading, LoRA patching, the training loop, checkpoint save/load and generation
all need a CUDA GPU and the gated Llama-3 weights. Smoke-test those on a GPU host:

```bash
python train_genolator.py --embedding_type pst \
    --output_dir ./smoke_out --hf_token "$HF_TOKEN" \
    --max_samples 512 --batch_size 2 --num_epochs 2 --patience 1
```

`--max_samples` caps the splits *after* loading, so the full training split is still
downloaded (~6.7 GiB); provision disk accordingly.

## Skips

Tests needing the Llama-3 tokenizer skip when it cannot be reached — the weights are
gated, so they need a cached copy or `HF_TOKEN` plus network. Everything else runs
offline against synthetic frames; no test downloads a dataset split.


