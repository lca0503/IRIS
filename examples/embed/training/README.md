# IRIS-Embed training

Train IRIS-Embed with ms-swift, then export a Hugging Face checkpoint.
Run commands from the repository root.

## Install dependencies

Use a separate environment with Python 3.12, PyTorch 2.11, and torchvision 0.26.

```bash
bash scripts/setup_training.sh
```

The script downloads ms-swift to `third_party/ms-swift`, applies the IRIS patch, and installs the training dependencies.
The ms-swift revision is fixed in the [patch files](../../../patches/ms-swift/README.md), and package versions are listed in [pyproject.toml](../../../pyproject.toml).
Qwen requires FlashAttention built for the installed PyTorch and CUDA versions.
Gemma uses SDPA and non-reentrant gradient checkpointing.
TorchCodec needs FFmpeg shared libraries to decode audio.

## Prepare data

Training and validation use JSONL files with these fields:

| Field | Content |
| --- | --- |
| `instruction` | Nonempty retrieval instruction |
| `spoken_query` | Query audio path |
| `positive_document` | Positive document audio path |
| `negative_documents` | List of negative document audio paths |
| `select_criteria` | Query criteria with at least one non-null value |
| `positive_document_metadata` | Positive document metadata for masking |
| `negative_documents_metadata` | Metadata for each negative, in the same order |
| `source` | Dataset source for grouping training rows |

Audio paths can be absolute or relative to the JSONL directory.
To use a shared audio directory for relative paths, set:

```bash
export IRIS_AUDIO_ROOT="$PWD/data/audio"
```

### Group training rows

The released models use grouped training data and validation data in its original order.
Create the grouped training file with:

```bash
bash examples/embed/training/group_data.sh data/train.jsonl data/train.grouped.jsonl
```

The script changes only row order.
It keeps rows with the same source and spoken query together, then packs them into source blocks of about 64 rows.
Query groups are never split, even when they exceed 64 rows.
Blocks are randomly ordered with seed 42, with no adjacent blocks from the same source.
The script stops if that arrangement is impossible or the output already exists.
Use `--block_size` or `--seed` to change these settings.
Save the grouped file beside the original JSONL, or set `IRIS_AUDIO_ROOT` so relative audio paths still resolve.

Training disables dataset and dataloader shuffling.
To train without grouping, pass `data/train.jsonl` instead.

## Train IRIS-Embed

Qwen:

```bash
NPROC_PER_NODE=16 bash examples/embed/training/train.sh \
  configs/embed/qwen25omni.json \
  data/train.grouped.jsonl data/val.jsonl outputs/iris-qwen \
  --audio_layer_aggregation learned \
  --audio_weight_backend minilm \
  --train_audio_projection true
```

Gemma:

```bash
NPROC_PER_NODE=16 bash examples/embed/training/train.sh \
  configs/embed/gemma4.json \
  data/train.grouped.jsonl data/val.jsonl outputs/iris-gemma \
  --audio_layer_aggregation learned \
  --audio_weight_backend minilm \
  --audio_weight_init_final_prob 0.9 \
  --train_audio_projection true
```

These settings train LLM LoRA, the MiniLM routing head, and the audio projection.
The pretrained MiniLM text encoder and audio encoder are frozen.
Qwen starts with nearly uniform layer weights.
Gemma starts with about 90% weight on the final audio layer.

### Multiple GPUs

The commands above use 16 GPUs on one machine, with 32 queries per GPU and one gradient accumulation step.
Each contrastive batch contains 512 queries and their positive and negative documents.
For multiple machines, set `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`, then run the command on each machine.
Four machines with four GPUs each use `NPROC_PER_NODE=4 NNODES=4`, node ranks 0–3, and the same address and port.
Changing the total GPU count changes the in-batch negative pool.
Gradient accumulation does not enlarge this pool.

## Audio encoder options

Options after the output directory override the config.
The configs default to the final audio layer and frozen projection.
All options below use LLM LoRA.

| Strategy | Options |
| --- | --- |
| Final layer | `--audio_layer_aggregation final` |
| Specific encoder layer | `--audio_layer_aggregation fixed --audio_layer_index 12` |
| Mean of all layers | `--audio_layer_aggregation mean` |
| Global layer weights | `--audio_layer_aggregation learned --audio_weight_backend global` |
| MiniLM layer weights | `--audio_layer_aggregation learned --audio_weight_backend minilm` |

Set `--train_audio_projection true` or `false` for any strategy.
Layer index 12 selects the output of encoder block 12, counting from zero.
Mean and learned mixtures include the pre-encoder state.
Softmax normalizes learned weights across layers.
Hidden states are not normalized before mixing.

### Train the full audio encoder

This trains the audio encoder and projection alongside LLM LoRA:

```bash
NPROC_PER_NODE=16 bash examples/embed/training/train.sh \
  configs/embed/qwen25omni.json \
  data/train.grouped.jsonl data/val.jsonl outputs/iris-qwen-full-audio \
  --audio_layer_aggregation final \
  --train_audio_projection true \
  --modules_to_save audio_tower \
  --vit_gradient_checkpointing true
```

For Gemma, use `configs/embed/gemma4.json` and `--modules_to_save model.audio_tower`.
Its projection outside the tower is saved when `--train_audio_projection true`.
The vision encoder stays frozen.

To freeze projection while training the encoder, set `--train_audio_projection false` and pass the matching `--freeze_parameters_regex`:

- Qwen: `'.*audio_tower\..*\.(ln_post|proj)\..*'`
- Gemma: `'.*audio_tower\..*\.output_proj\..*'`

The regex freezes projection parameters inside the saved audio tower.

## Loss and training settings

Queries use `Instruction: <instruction>\nQuery: <audio>` and documents use `Document: <audio>`.
No system prompt is added.
MiniLM receives the text prefixes without audio.
The last valid LLM hidden state is L2-normalized before the loss.

Training uses query-to-document InfoNCE with temperature 0.03 and in-batch documents from all devices.
Metadata masking removes candidate negatives that satisfy all active query criteria, while keeping the query's own positive.
Cross-source speaker masking is disabled.

| Setting | Value |
| --- | --- |
| Optimizer | Fused AdamW, LR 2e-4, weight decay 0.1 |
| Schedule | Cosine decay, 5% warmup, 2 epochs |
| LoRA | LLM linear layers, rank 64, alpha 128, dropout 0.05 |
| Validation and saving | Every 200 steps |
| Checkpoints kept | Up to 18, including optimizer state |
| Checkpoint selection | Final checkpoint; `load_best_model_at_end` is false |

Validation reports InfoNCE loss.
To resume training, add `--resume_from_checkpoint outputs/iris-qwen/checkpoint-STEP` to the original command with the same data and model options.

## Export a checkpoint

```bash
bash examples/embed/training/merge.sh \
  outputs/iris-qwen/checkpoint-STEP checkpoints/iris-qwen data/query.wav
```

Replace `checkpoint-STEP` with the checkpoint to export and use a new output directory.
The script merges LoRA and saved audio modules, copies the model and processor code, restores the training chat template, and checks the exported files.
MiniLM checkpoints also include the instruction tokenizer.
The final audio argument is optional and tests query and document encoding.

Use the exported directory with the [inference examples](../inference/README.md), [INSPIRE experiments](../experiments/inspire/README.md), or [MAEB evaluation](../experiments/maeb/README.md).
