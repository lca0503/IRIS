# MAEB evaluation

Evaluate IRIS-Embed on 15 MAEB tasks with MTEB 2.18.0.
The model stays frozen, and each task follows MTEB's data splits and evaluation protocol.
Run commands from the repository root.

## Install dependencies

Use a separate environment with Python 3.12, PyTorch 2.11, torchvision 0.26, and torchaudio 2.11.

```bash
python -m pip install -e '.[maeb]'
```

TorchCodec needs FFmpeg shared libraries to decode audio.

## Evaluate a checkpoint

Qwen:

```bash
bash examples/embed/experiments/maeb/evaluate.sh \
  lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  results/MAEB/qwen/query
```

Gemma:

```bash
bash examples/embed/experiments/maeb/evaluate.sh \
  lca0503/IRIS-Embed-gemma-4-E2B-it \
  results/MAEB/gemma/query
```

Both commands run all 15 tasks with query prompting by default.
You can replace the model ID with a local exported checkpoint directory.
Use a separate output directory for each model and setting.
MTEB downloads task data through the Hugging Face cache.
Some datasets require access approval or authentication.

### Select tasks

```bash
bash examples/embed/experiments/maeb/evaluate.sh \
  lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  results/MAEB/qwen/selected_tasks \
  --tasks IEMOCAPGender MInDS14 NMSQAPairClassification \
  --batch_size 4
```

Add options after the output directory:

| Option | Purpose | Default |
| --- | --- | --- |
| `--tasks` | Task names from the table below | All 15 tasks |
| `--batch_size` | Recordings per batch | 16 |
| `--embedding_side` | Prompt for classification, clustering, and pair classification | `query` |
| `--device` | Device, such as `cuda:0` or `cpu` | `auto` |
| `--revision` | Hugging Face model revision | Default model revision |
| `--label` | Model name in reports | Checkpoint name |

Use `--embedding_side document` to test document prompts, with a new output directory.
`--embedding_side auto` uses a document prompt when MTEB requests one and a query prompt otherwise.
GTZAN reranking and JamAlt retrieval always use query prompts for queries and `Document:` for candidates.
Pair classification uses the selected prompt for both recordings in each pair.

## Compare results

```bash
python src/iris/embed/experiments/maeb/report.py \
  --run_dirs results/MAEB/qwen/query results/MAEB/gemma/query \
  --csv_file results/MAEB/reports/query.csv
```

The report prints a table and saves the CSV.
Omit `--csv_file` to print only, or add `--tasks` to choose columns.
Each model and prompt setting has its own row.
Reports require completed runs and leave unevaluated tasks blank.
Scores use each task's main metric on MTEB's original scale.
The report does not average across different tasks.

## Resume a run

Repeat the original command with `--resume`:

```bash
bash examples/embed/experiments/maeb/evaluate.sh \
  lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  results/MAEB/qwen/query \
  --resume
```

Resume reuses completed tasks.
Keep the checkpoint, revision, prompt, task selection, batch size, label, and package versions the same.
If using a local checkpoint, keep its files unchanged.
Use a new output directory to change settings.

## Tasks and instructions

| Task | Query instruction | Main metric |
| --- | --- | --- |
| BeijingOpera | Find recordings with the same percussion instrument as the query. | Accuracy |
| BirdCLEF | Find recordings with the same bird species as the query. | Accuracy |
| CommonLanguageAgeDetection | Find recordings with the same speaker age group as the query. | Accuracy |
| GTZANGenre | Find recordings with the same music genre as the query. | Accuracy |
| IEMOCAPGender | Find recordings with the same gender as the query. | Accuracy |
| MInDS14 | Find recordings with the same intent as the query. | Accuracy |
| MridinghamTonic | Find recordings with the same tonic as the query. | Accuracy |
| SIBFLEURS | Find recordings with the same topic as the query. | Accuracy |
| VoxPopuliLanguageID | Find recordings with the same spoken language as the query. | Accuracy |
| VehicleSoundClustering | Find recordings with the same environmental sound as the query. | V-measure |
| VoxPopuliGenderClustering | Find recordings with the same gender as the query. | V-measure |
| NMSQAPairClassification | Find recordings that contain the answer to the query. | Max AP |
| VoxPopuliAccentPairClassification | Find recordings with the same accent as the query. | Max AP |
| GTZANAudioReranking | Find recordings with the same music genre as the query. | MAP@1000 |
| JamAltArtistA2ARetrieval | Find recordings by the same artist as the query. | NDCG@10 |

Queries use `Instruction: <instruction>\nQuery: <audio>` and documents use `Document: <audio>`.
MiniLM receives `Instruction: <instruction>\nQuery:` or `Document:` without audio.
The checkpoint's chat template adds the assistant prefix, with no system or summarization message.

## Evaluation details

Audio is converted to mono, resampled to the model's sampling rate, and cut to the first 120 seconds.
The 120-second limit applies only to MAEB, not the inference API.
Embeddings use the last non-padding token of the final LLM layer and are returned as L2-normalized float32 arrays.
No centering or calibration is applied.

Task-specific classifiers follow MTEB's protocol while IRIS stays frozen.
These task splits do not select an IRIS checkpoint.
MTEB's `TaskResult.get_score()` combines each task's scores across its evaluated subsets and splits.

## Saved files

```text
<output_dir>/
  run.json      Model, tasks, instructions, settings, versions, and completion
  summary.json  One main score per task
  mteb/         MTEB task results and model metadata
```

`run.json` also records each task's dataset revision, evaluation splits, metric, and seed.
A run is complete when every selected task has a finite score.
