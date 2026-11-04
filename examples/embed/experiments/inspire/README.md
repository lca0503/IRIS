# INSPIRE experiments

Evaluate retrieval on INSPIRE and compare results across models and query instructions.
Run commands from the repository root.

## Install dependencies

Use the versions listed in the [inference guide](../../inference/README.md#install-dependencies).

```bash
python -m pip install -e '.[experiments]'
```

TorchCodec needs FFmpeg shared libraries to decode audio.

## Evaluate a checkpoint

These commands encode all four [INSPIRE](https://huggingface.co/datasets/lca0503/INSPIRE) splits and calculate Recall and NDCG.

Qwen:

```bash
bash examples/embed/experiments/inspire/evaluate.sh \
  lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  results/INSPIRE/models/IRIS-Embed-Qwen2.5-Omni-3B/general
```

Gemma:

```bash
bash examples/embed/experiments/inspire/evaluate.sh \
  lca0503/IRIS-Embed-gemma-4-E2B-it \
  results/INSPIRE/models/IRIS-Embed-gemma-4-E2B-it/general
```

You can replace the model ID with a local exported checkpoint directory.
Use a new or empty output directory for each evaluation or instruction experiment.
If encoding stops before completion, restart in a new directory.

Add encoding options after the output directory, such as `--splits DailyTalk VCTK --device cuda:0`.
Use `--revision` for the model revision and `--dataset_revision` for the dataset revision.
Settings are saved in `run.json`.

## Compare checkpoint scores

```bash
python src/iris/embed/experiments/inspire/report.py \
  --run_dirs \
    results/INSPIRE/models/IRIS-Embed-Qwen2.5-Omni-3B/general \
    results/INSPIRE/models/IRIS-Embed-gemma-4-E2B-it/general \
  --group_by splits \
  --metrics Recall@10 Recall@50 Recall@100 \
  --csv_file results/INSPIRE/reports/by_split_recall.csv
```

For relevance results, use `--group_by relevance --metrics Recall@50` and a different CSV filename.
`--run_dirs` sets which models to compare and their row order.
Omit `--csv_file` to print results without saving a CSV.
`--metrics` accepts `Recall@k`, `R@k`, and `NDCG@k`.
The defaults are Recall and NDCG at 10, 50, and 100.

Columns follow DailyTalk, VCTK, Expresso, Synthetic, with scores from 0 to 1.
Relevance CSVs have three header rows: split, relevance, and metric.
Named target styles and sounds are grouped as specific style or sound.
Missing splits are left blank.

## Instruction experiments

First finish encoding a base evaluation run.
Scoring the base run is not required for these experiments.
The scripts load its checkpoint and re-encode only queries with new instructions.
They reuse the document embeddings, positive documents, and exclusions from the base run.

The examples use Qwen.
For Gemma, replace `IRIS-Embed-Qwen2.5-Omni-3B` with `IRIS-Embed-gemma-4-E2B-it` in both paths.

### Short, normal, and incorrect instructions

```bash
bash examples/embed/experiments/inspire/instruction_ablation.sh \
  results/INSPIRE/models/IRIS-Embed-Qwen2.5-Omni-3B/general \
  results/analysis/instruction_ablation/IRIS-Embed-Qwen2.5-Omni-3B/general
```

Each query is encoded with all ten instructions below and the empty instruction `""`, giving 11 conditions.

| Task | Short instruction | Normal instruction |
| --- | --- | --- |
| Continuation | Continuation. | Find speech that continues the conversation in the query. |
| Speaker | Same speaker. | Find speech from the same speaker as the query. |
| Speaking style | Same speaking style. | Find speech with the same speaking style as the query. |
| Environmental sound | Same environmental sound. | Find speech with the same environmental sound as the query. |
| Answer retrieval | Find the answer. | Find speech that contains the answer to the question in the query. |

For the empty instruction, the query prompt is `Instruction: \nQuery: <audio>` and MiniLM receives `Instruction: \nQuery:`.
Score rows keep the dataset instruction in `original_instruction`.
Use the base evaluation for scores under the original instructions.

### Instruction composition

```bash
bash examples/embed/experiments/inspire/composed_instructions.sh \
  results/INSPIRE/models/IRIS-Embed-Qwen2.5-Omni-3B/general \
  results/analysis/instruction_ablation/IRIS-Embed-Qwen2.5-Omni-3B/synthetic_composed
```

This experiment selects four Synthetic relevance groups:

| Relevance group | Composed instruction |
| --- | --- |
| same speaker and same environmental sound | Same speaker and same environmental sound. |
| same speaker and same speaking style | Same speaker and same speaking style. |
| same speaker and same speaking style and same environmental sound | Same speaker and same speaking style and same environmental sound. |
| same speaking style and same environmental sound | Same speaking style and same environmental sound. |

Each group uses all four composed instructions, the five short instructions above, and the empty instruction.
This gives 40 conditions per model.
Candidates include all Synthetic documents except each query's excluded documents.

### Custom instructions

Compare two instructions on Synthetic answer-retrieval queries:

```bash
python src/iris/embed/experiments/inspire/instruction_ablation.py \
  --base_run_dir results/INSPIRE/models/IRIS-Embed-Qwen2.5-Omni-3B/general \
  --splits Synthetic \
  --relevances 'contain answer' \
  --instructions 'Find the answer.' 'Same speaker.' \
  --output_dir results/analysis/instruction_ablation/qwen_answer_vs_speaker
```

`--relevances` takes exact dataset labels.
Omit it to include all queries, or pass several labels to evaluate them separately.
Use `--instructions ""` to test only the empty instruction.
The shell scripts also accept `--device`, `--k_values`, and a replacement `--instructions` list after the output directory.

### Compare instruction scores

```bash
python src/iris/embed/experiments/inspire/report_instructions.py \
  --run_dirs \
    results/analysis/instruction_ablation/IRIS-Embed-Qwen2.5-Omni-3B/general \
    results/analysis/instruction_ablation/IRIS-Embed-gemma-4-E2B-it/general \
  --group_by relevance \
  --metrics Recall@50 \
  --csv_file results/INSPIRE/reports/instruction_ablation_recall50.csv
```

Include only completed runs in `--run_dirs`.
For composition results, replace `general` with `synthetic_composed` in the analysis paths.
Each row shows the model, relevance filter, and exact instruction.
The empty instruction has its own row.
Use `--instructions` or `--relevances` to select exact strings from the results.
Grouping, metric, and CSV options work the same way as in the checkpoint report.

## Score saved embeddings

The default k values are 1, 5, 10, 20, 50, and 100.
To replace a base run's scores using different k values:

```bash
python src/iris/embed/experiments/inspire/score.py \
  --run_dir results/INSPIRE/models/IRIS-Embed-Qwen2.5-Omni-3B/general \
  --k_values 10 50 100
```

Scoring uses the saved embeddings and runs on CPU.

## Scoring details

Queries use the dataset instruction or its replacement, while documents use `Document:`.
Both use the checkpoint's chat template and assistant prefix, with no system or summarization message.
Encoding processes one recording at a time and saves the unnormalized last valid token from the final LLM layer.
Retrieval ranks documents in the query's split by cosine similarity after removing `excluded_ids`.
Recall@k is the fraction of positive documents retrieved in the top k.
NDCG@k uses binary relevance and logarithmic rank discounts.
Reports average query scores within each split or `(split, relevance)` group.

The default seed for breaking exact score ties is 42.
Base evaluation keeps one random stream across queries, with documents in cache file order.
Instruction experiments reset a split-specific random stream for each condition, with documents in dataset order.

## Saved files

Base evaluation:

```text
<run>/
  run.json
  representations/queries/<split>/00000.npy
  representations/documents/<split>/<audio_id>.npy
  scores/<split>/score.jsonl
```

Instruction experiments:

```text
<analysis>/
  run.json
  <relevance-or-all>/<instruction>/
    run.json
    representations/queries/<split>/00000.npy
    scores/<split>/score.jsonl
```

Each instruction condition records its exact instruction, original query row indices, and base run.
Document embeddings stay in the base run.
Reports require completed scoring runs.

See [layer weight plots](layer_weights.md) for raw and RMS-adjusted router heatmaps.
That analysis loads the checkpoint directly and does not need saved evaluation embeddings.
