# IRIS-Embed inference

Encode query and document audio, then compare the embeddings with cosine similarity.
Run commands from the repository root and replace the example audio paths with your files.

## Install dependencies

Use Python 3.12, PyTorch 2.11, and torchvision 0.26.

```bash
python -m pip install -e .
```

## Choose a checkpoint

| Model | Hugging Face checkpoint | Embedding dimensions |
| --- | --- | --- |
| Qwen | [lca0503/IRIS-Embed-Qwen2.5-Omni-3B](https://huggingface.co/lca0503/IRIS-Embed-Qwen2.5-Omni-3B) | 2048 |
| Gemma | [lca0503/IRIS-Embed-gemma-4-E2B-it](https://huggingface.co/lca0503/IRIS-Embed-gemma-4-E2B-it) | 1536 |

The examples use Qwen.
For Gemma, set `--model_name lca0503/IRIS-Embed-gemma-4-E2B-it` and use separate output files.
You can also pass a local checkpoint exported with the [training guide](../training/README.md).
Use the same checkpoint for queries and documents.

## Encode queries

Queries require a nonempty instruction describing what to retrieve.
The command uses the same instruction for every query audio.

```bash
python examples/embed/inference/encode.py \
  --model_name lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  --audio data/query.wav \
  --role query \
  --instruction "Same speaker." \
  --output_file results/qwen/query_embeddings.npy
```

## Encode documents

Omit `--instruction` for documents.

```bash
python examples/embed/inference/encode.py \
  --model_name lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  --audio data/document1.wav data/document2.wav \
  --role document \
  --output_file results/qwen/document_embeddings.npy
```

The `.npy` files contain L2-normalized float32 arrays with shape `(number of recordings, embedding dimensions)`.
Rows follow the order of `--audio`.
Use a new filename for each output.

## Retrieve documents

```bash
python examples/embed/inference/retrieve.py \
  --model_name lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  --query data/query.wav \
  --documents data/document1.wav data/document2.wav \
  --instruction "Same speaker." \
  --top_k 2
```

The command encodes the audio and prints each document's rank, cosine similarity, and path.
For saved normalized embeddings, calculate cosine scores with `queries @ documents.T`.
You can reuse document embeddings for other queries and instructions from the same checkpoint.

## Input and embedding details

Use `--revision` to select a model revision.
`--device` accepts `auto`, `cpu`, `cuda`, or `cuda:0` and defaults to `auto`.
Audio files are converted to mono and resampled to the model's sampling rate.
Encoding processes one recording at a time and does not truncate audio.

The query prompt is:

```text
Instruction: <instruction>
Query: <audio>
```

Documents use `Document: <audio>`.
The checkpoint's chat template adds the assistant prefix, with no system or summarization message.
The embedding comes from the last non-padding token of the final LLM layer.
MiniLM receives only the text `Instruction: <instruction>\nQuery:` or `Document:`.

For Python usage, see the [quick start](../../../README.md#quick-start).
The API accepts file paths or mono audio mappings with `array` and `sampling_rate` fields.
`encode_queries` takes one shared instruction or one instruction per recording.
Pass `normalize=False` to get unnormalized embeddings.
The model loader also accepts `dtype` and `attn_implementation`.

To evaluate a benchmark, follow the [INSPIRE](../experiments/inspire/README.md) or [MAEB](../experiments/maeb/README.md) guide.
