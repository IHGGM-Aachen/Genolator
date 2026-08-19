# Genolator V1 — Publication Release

Training and inference code for the two Genolator models reported in the paper.
Both are Llama-3 models extended with **virtual token projectors** that map
precomputed genomic embeddings into the language model's input embedding space,
so the LLM can answer gene-level questions conditioned on sequence and structure.

| Published model | Modalities | Checkpoint | Weights |
| --- | --- | --- | --- |
| `genolator_v1_dna_and_pst` | DNA (Evo2) + protein structure (PST) | `genolator_dna_and_pst.pt` | [`Llama-3-Genolator-v1-PST`][pst] |
| `genolator_v1_dna_and_esm` | DNA (Evo2) + amino acid (ESM-2) | `genolator_dna_and_esm.pt` | [`Llama-3-Genolator-v1-ESM`][esm] |

[pst]: https://huggingface.co/CHGGM-Aachen/Llama-3-Genolator-v1-PST
[esm]: https://huggingface.co/CHGGM-Aachen/Llama-3-Genolator-v1-ESM

Each published model is a directory holding three files: the checkpoint above,
`dna_projector.pt` and the protein-side projector (`esm_projector.pt` or `pst_projector.pt`). The directory name itself
carries no meaning -- inference takes the three paths as explicit arguments.

The two variants share one architecture and one training procedure; they differ
only in which protein-side embedding column they consume. Both are therefore
driven by a single training script and a single inference script, selected with
`--embedding_type {pst,esm}`.

## Releases

Everything needed to reproduce or reuse the models is on the Hugging Face Hub under the
[CHGGM-Aachen](https://huggingface.co/CHGGM-Aachen) organisation:

| | Repository |
| --- | --- |
| Dataset | [`CHGGM-Aachen/genolator-v1-qa`](https://huggingface.co/datasets/CHGGM-Aachen/genolator-v1-qa) |
| DNA + PST model | [`CHGGM-Aachen/Llama-3-Genolator-v1-PST`][pst] |
| DNA + ESM model | [`CHGGM-Aachen/Llama-3-Genolator-v1-ESM`][esm] |

Each model repository holds the checkpoint, both projectors and a training summary, with a
model card covering the exact inference flags. The dataset repository holds the
gene-disjoint train/validation/test splits with the precomputed embeddings these scripts
consume. See [Dataset](#dataset) for how the code reads them.

## Contents

```
genolator_release/
├── train_genolator.py                      # train either model  (--embedding_type)
├── run_inference.py                        # generate with either model
├── utils/
│   ├── columns.py                          # dataset column names (edit here to remap)
│   ├── data.py                             # split loading: HF Hub or local files
│   ├── token_projector.py                  # GenomicVirtualTokenProjector
│   ├── qa_dataset.py                       # GenomicQADatasetBase + {ESM,PST} variants
│   └── utils.py                            # collate, evaluation, generation loops
├── examples/                               # runnable reproductions of both models
├── tests/                                  # CPU-only unit tests (uv sync --extra test)
├── pyproject.toml                          # dependencies (uv sync [--extra train])
├── uv.lock                                 # exact pinned resolution
├── .python-version                         # CPython 3.11
├── requirements.txt                        # pip fallback, same constraints
└── LICENSE
```

## Installation

Everything is managed with [uv](https://docs.astral.sh/uv/). Dependencies are
declared in `pyproject.toml` and pinned in `uv.lock`, so the environment is one
command:

```bash
uv sync --extra train     # training and inference
uv sync                   # inference only, no unsloth
```

That creates `.venv/`, fetches CPython 3.11 (pinned in `.python-version`) if it
is not already available, and installs the locked versions. If you do not have
uv yet: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Nothing is installed as an importable package -- `[tool.uv] package = false`, so
uv only manages the environment and the two entrypoints stay plain scripts:

```bash
uv run python train_genolator.py --embedding_type pst ...
uv run python run_inference.py  --embedding_type pst ...
uv run ./examples/train_pst.sh          # the example wrappers work too
```

`uv run` finds the project root from any subdirectory, so the `cd examples`
workflow below is unaffected. Alternatively `source .venv/bin/activate` once and
call `python3` directly. Use `uv sync --frozen` to install the lock exactly as
committed and fail rather than re-resolve, and `uv lock` to deliberately refresh
it.

`unsloth` is a `train` extra rather than a base dependency for two reasons:
inference rebuilds the identical model with stock `transformers` + `peft` and
never imports it, and it resolves only on CUDA platforms -- so a plain
`uv sync` still works on a laptop for reading and running inference code paths.
For an unusual CUDA/torch combination, install it following the
[unsloth instructions](https://github.com/unslothai/unsloth) instead of letting
the plain wheel resolve.

<details>
<summary>Without uv (pip)</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes unsloth
```

`requirements.txt` carries the same version constraints as `pyproject.toml` but
no lockfile, so resolution is up to pip.
</details>

**Version constraints.** Each dependency has an upper bound on its major
version. These matter: the code targets the `transformers` 4.x API, and an
unbounded resolve today selects `transformers` 5.x, which installs cleanly and
then fails at runtime. `uv.lock` was resolved on 2026-08-17 and selects
`torch` 2.13, `transformers` 4.57, `peft` 0.20 and `unsloth` 2026.3. It is a
known-good resolution rather than a reconstruction of the environment behind
the published numbers: those packages were installed unpinned at the time, so
the exact versions were never recorded.

Python 3.11 and a CUDA GPU are required. Both models were trained and evaluated
on a single H100 80GB in BF16 (no 4-bit quantization).

The base model `ContactDoctor/Bio-Medical-Llama-3-8B` is gated, so a HuggingFace
token is required:

```bash
export HF_TOKEN=hf_your_token_here
```

## Dataset

The train/validation/test splits used for the published models are released on the
HuggingFace Hub as [`CHGGM-Aachen/genolator-v1-qa`](https://huggingface.co/datasets/CHGGM-Aachen/genolator-v1-qa).
Both entrypoints read them from there by default — there is nothing to download by
hand, and no splitting step: the splits are published gene-disjoint and the training
split is already class-balanced per `group`.

```bash
# uses CHGGM-Aachen/genolator-v1-qa
python train_genolator.py --embedding_type pst --output_dir ./out --hf_token "$HF_TOKEN"
```

`huggingface_hub` downloads each split's parquet file once and caches it under
`HF_HOME` (`~/.cache/huggingface` by default), so later runs start immediately. Set
`HF_HOME` to move that cache onto a volume with room for it.

`--dataset` takes any dataset repo id, or a local directory holding the splits.

The trained weights that go with these splits are linked under [Releases](#releases).
`--dataset_revision` pins a branch, tag or commit sha, which is what makes a run
exactly reproducible against a dataset that may later change.

To train or evaluate on your own data, override an individual split with a local
file — either a `.parquet` file or a pickled DataFrame:

```bash
python train_genolator.py --embedding_type pst \
    --train_path /data/my_train.parquet \
    --val_path   /data/my_val.parquet \
    --test_path  /data/my_test.parquet \
    --output_dir ./out --hf_token "$HF_TOKEN"

python run_inference.py --embedding_type pst --dataset_path /data/my_test.parquet ...
```

A directory passed to `--dataset` is searched for `<split>.parquet`,
`data/<split>.parquet` and `<split>.pkl`, so a directory produced by
`huggingface-cli download ... --local-dir ./data` works as-is.

### Columns

| Column | Used by | Description |
| --- | --- | --- |
| `cdna_seq_embedding` | both | Evo2 cDNA embedding (4096-d) |
| `pst_embedding` | PST model | Protein structure embedding (1280-d) |
| `aa_seq_embedding` | ESM model | ESM-2 amino acid embedding (2560-d) |
| `prompt` | both | Question about the gene |
| `response` | both | Target answer |
| `kind` | both | Question type; training keeps `confirmation`, `denial`, `generic` |
| `group` | training | Subset id in `{1, 2, 3}` for epoch-wise rotation; train split only |
| `gene_name`, `go_aspect` | inference | Carried into the results table |

Parquet reads are column-pruned to exactly this list, minus the modality that is not
selected. The dataset also carries `dna_seq`, `cdna_seq` and `aa_seq`, which no run
reads. Of the 6.68 GiB train split, a PST run decodes 3.85 GiB and an ESM run
4.04 GiB; the rest — the unselected modality's embeddings (the larger part) and the
raw sequence columns (15%) — is skipped. Note that pruning saves decode time and
memory, not download: `huggingface_hub` fetches the whole parquet file either way.
Pickled DataFrames cannot be pruned and are read whole.

Embedding columns come back as 1-D float32 numpy arrays. Projector input dimensions
are read from the first row at runtime, so the embedding widths do not need to be
configured.

If your DataFrame names the embedding columns differently, change them in
`utils/columns.py` — that is the only place the three embedding columns are
defined, and both entrypoints, the datasets and the generation loop read them
from there. Split resolution and the per-run column lists live in `utils/data.py`.

## Training

```bash
cd examples
export HF_TOKEN=hf_your_token_here
./train_pst.sh                             # or ./train_esm.sh

# or against local splits instead of the HuggingFace dataset:
DATA_DIR=/path/to/splits ./train_pst.sh
```

Both wrap `train_genolator.py --embedding_type {pst,esm}` and carry the exact
hyperparameters of the published runs:

| Setting | Value |
| --- | --- |
| Base model | `ContactDoctor/Bio-Medical-Llama-3-8B`, max seq length 8192, BF16 |
| Virtual tokens | 8 per modality (16 prepended in total) |
| Text tokenization | `max_length=256` (dataset default; not exposed as a flag) |
| LoRA | `r=8`, `alpha=32`, `dropout=0.05` |
| LoRA target modules | `q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj` (`--lora_all_target_modules true`) |
| Virtual embedding scaling | enabled (`--scale_virtual_embeddings true`) |
| Batch size | 8 |
| Epochs | 10, early stopping on validation loss with patience 3 |
| Optimizer | AdamW, LR `5e-5` for the LoRA and both projector parameter groups |

Training rotates over the `group` column (`group = epoch % 3 + 1`) so each epoch
sees a balanced cluster subset, computes next-token loss over the answer only
(virtual token positions are masked with `-100`), and writes the best-validation
checkpoint.

**Reproducibility note:** `--learning_rate` defaults to `5e-5` and drives the LoRA
and both projector parameter groups, which is what the published runs used.
`--projector_learning_rate` overrides the rate for the two projectors only; leave
it unset to reproduce the published configuration.

**Loss over the answer only:** padding is masked out of the labels. Because `pad_token` is set
to `eos_token`, earlier versions of this code left every padding position supervised. Those
positions are trivial to predict, so they diluted the loss, left less training signal per step
and cost training efficiency. The first pad position is kept as the stop token, since the
tokenizer appends no EOS of its own and generation stops on EOS; everything after it is `-100`.
Losses printed by this code are therefore higher than, and not comparable to, the ones reported
for the published checkpoints. Those checkpoints and the Zenodo snapshot were trained before
this change, so the figures reported for them stay reproducible from those artefacts.

### Smoke test

`--max_samples` caps every split so a run finishes in minutes instead of days,
which is the quickest way to check that a machine can train at all before
committing to the full dataset:

```bash
python train_genolator.py --embedding_type pst --hf_token $HF_TOKEN \
    --max_samples 32 --num_epochs 1 --output_dir /tmp/genolator_smoke
```

The cap applies per split. The training split is capped per `group` so the epoch
rotation still sees data from every group. Leave the flag unset for real runs;
it is logged to MLflow so a capped run is never mistaken for a full one.

The tests in `tests/` cover the CLI, split subsetting, dataset/collate shapes and
the projector on CPU, with no GPU, no model download and no dataset download:

```bash
uv sync --extra test
uv run pytest tests/
```

### Outputs

Written to `--output_dir`:

- `genolator_dna_and_pst.pt` / `genolator_dna_and_esm.pt` — full model state dict
- `dna_projector.pt` — DNA virtual token projector
- `pst_projector.pt` / `esm_projector.pt` — protein-side projector
- `config.json` — embedding dims, virtual token count, best validation loss, epoch

Metrics and hyperparameters are logged to MLflow. **No MLflow server is
required:** the code never sets a tracking URI, so MLflow falls back to a local
store in the current working directory (`./mlflow.db` with MLflow 3.x,
`./mlruns` with 2.x) and training runs offline. Point it at a server by setting
`MLFLOW_TRACKING_URI` if you want one -- but note that if that variable names a
server that is not reachable, the run blocks on connection retries rather than
failing fast, so leave it unset when working offline.

Nothing else in the code is tied to a particular platform -- it reads and writes
plain local paths, so it runs unchanged on-prem or on any cloud.

## Inference

```bash
cd examples
export HF_TOKEN=hf_your_token_here
CHECKPOINT_DIR=/path/to/genolator_v1_dna_and_pst \
  ./inference_pst.sh      # or ./inference_esm.sh

# or against a local test split instead of the HuggingFace dataset:
DATA_DIR=/path/to/splits \
CHECKPOINT_DIR=/path/to/genolator_v1_dna_and_pst \
  ./inference_pst.sh
```

The evaluated split defaults to `test`; `--split {train,validation,test}` selects
another one.

The LoRA and virtual-token flags must match training, otherwise the state dict
will not load onto the reconstructed model. The example scripts already pass the
published configuration.

Inference streams the dataset in `--subset_size` chunks (200 in the published
runs), writes one pickle per chunk into `--output_dir`, then concatenates them
into `inference_genolator_v1_unsloth_{pst,esm}_{output_suffix}.pkl`.

Result columns: `gene_name`, `go_aspect`, `prompt`, `response`, `generated_text`,
`kind`, plus `hidden_states` and the attention traces `dna_attn`,
`pst_attn`/`esm_attn`, `prompt_attn`, `output_attn`.

### Evaluation subset (`--kinds`)

`--kinds` restricts evaluation to the listed question kinds; omitting it
evaluates the full dataset. The selected subset and the resulting row count are
logged to MLflow as `eval_kinds` and `eval_num_samples`.

**The two published runs did not use the same evaluation subset:**

| Published run | Effective selection | Reproduced by |
| --- | --- | --- |
| PST | full test set (no filtering) | `--kinds` omitted — see `examples/inference_pst.sh` |
| ESM | `confirmation` and `denial` only (`generic` excluded) | `--kinds confirmation denial` — see `examples/inference_esm.sh` |

The example scripts reproduce each published run exactly. If you intend to
compare PST against ESM, pass the same `--kinds` to both; the numbers reported
for the two models were computed over different subsets of the test set.

## Architecture

```
DNA embedding      (dim D) ──► Linear ──► 8 virtual tokens (8 × H)
Protein embedding  (dim P) ──► Linear ──► 8 virtual tokens (8 × H)
Prompt + answer            ──► embedding lookup ──► T text tokens (T × H)

LLaMA input: [ 8 DNA vtokens | 8 protein vtokens | T text tokens ]   (H = 4096)
```

Each projector is a single `Linear(embedding_dim → 8 × H)` with dropout and a
learnable `null_emb` fallback used when a modality is missing for a sample. With
`--scale_virtual_embeddings true` the virtual tokens are rescaled to match the
mean L2 norm of the text token embeddings in the batch, which keeps the injected
modalities on the same scale as the language tokens.

## Press

Genolator is featured in a Microsoft customer success story on Universitätsklinikum
Aachen, which includes a video:
<https://www.microsoft.com/en/customers/story/26455-universitatsklinikum-aachen-aor-azure>

## License

See `LICENSE`.

## Contact

mdanner@ukaachen.de
