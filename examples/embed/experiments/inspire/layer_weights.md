# Instruction layer weights

Extract the router's layer weights and measure audio-layer scales with either
released checkpoint. Run from the repository root after installing the
`experiments` extra:

```bash
bash examples/embed/experiments/inspire/layer_weights.sh \
  lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  results/analysis/layer_weights/qwen

bash examples/embed/experiments/inspire/layer_weights.sh \
  lca0503/IRIS-Embed-gemma-4-E2B-it \
  results/analysis/layer_weights/gemma
```

The same command accepts a local exported IRIS checkpoint. Use a new output
directory for each run. `--revision` selects the model revision; `--device`
defaults to `auto`. This analysis supports MiniLM and global weighting.

The script loads the checkpoint through `IRISEmbedder`, then runs only the
router and audio encoder. It does not run the LLM, compute retrieval scores, or
change checkpoint parameters. Loading still requires memory for the complete
checkpoint. Audio inference uses its saved precision; `--device cpu` uses
float32. The raw router calculation always uses a separate float32 CPU copy,
matching the paper analysis.

## Instructions and audio

Every CSV includes these ten query instructions and the document condition:

| Criterion | Short instruction | Full instruction |
| --- | --- | --- |
| Contain answer | Find the answer. | Find speech that contains the answer to the question in the query. |
| Continuation | Continuation. | Find speech that continues the conversation in the query. |
| Same speaker | Same speaker. | Find speech from the same speaker as the query. |
| Same speaking style | Same speaking style. | Find speech with the same speaking style as the query. |
| Same environmental sound | Same environmental sound. | Find speech with the same environmental sound as the query. |

Query routing text is `Instruction: <instruction>\nQuery:`; document routing
text is `Document:`. Raw weights depend on this text, so each condition needs
only one router forward.

For RMS adjustment, use all **680 unique query recordings** in the pinned
INSPIRE revision: DailyTalk 200, VCTK 80, Expresso 200, Synthetic 200. Repeated
`audio_id` values within a split keep their first dataset row. Every instruction
and the document condition use this same audio pool. The document row therefore
measures document routing weights on the shared query audio pool; it does not
sample the document corpus. No random audio sampling is involved.

The default dataset revision is
`4202b965891ca79897ce8cb639e08c591e993c06`. An explicit `--dataset_revision`
must retain these split counts. The exact selected rows are saved in
`samples.jsonl`.

## RMS adjustment

Capture `pre_encoder` and every encoder block output **before layer mixing or
projection**. For each audio and layer, compute channel RMS separately at each
valid frame, then average those RMS values across frames. Padding is excluded.

For each audio, multiply each raw weight by that layer's RMS and normalize the
products to sum to one. Average these normalized vectors within each split,
then average the four split vectors equally. Thus VCTK has the same overall
weight as each other split, despite having fewer recordings. Each recording
also receives equal weight within its split, regardless of duration.

This is a scale-adjusted description of the mixture, not a causal measure of
layer importance: it does not account for alignment or cancellation between
layers. It does not add normalization to the model or change its actual weights.

## Outputs

```text
<output_dir>/
  run.json                              Model, dataset, precision, and analysis settings
  layer_weights.csv                     Raw weights: ten query rows plus Document
  samples.jsonl                         All 680 selected recordings
  scale/per_audio_rms.csv                Mean frame RMS for each recording and layer
  layer_weights_rms_adjusted.csv         Equal-split average; num_audio = 680
  layer_weights_rms_adjusted_by_split.csv Per-split averages and recording counts
  plots/raw.png                         Raw heatmap
  plots/raw.pdf
  plots/rms_adjusted.png                 RMS-adjusted heatmap
  plots/rms_adjusted.pdf
  plots/settings.json                   Plot settings and color limits
```

Only CSV statistics are retained for audio scales; per-frame arrays are not
saved. Each checkpoint measures its own encoder scales.

To get only raw weights and the raw plot, without loading INSPIRE audio:

```bash
bash examples/embed/experiments/inspire/layer_weights.sh \
  lca0503/IRIS-Embed-Qwen2.5-Omni-3B \
  results/analysis/layer_weights/qwen_raw \
  --weights_only --device cpu
```

## Replot saved CSVs

```bash
python src/iris/embed/experiments/inspire/plot_layer_weights.py \
  --run_dirs results/analysis/layer_weights/qwen results/analysis/layer_weights/gemma \
  --font_size 14
```

Plotting does not load a model or audio. By default, each 8 × 4 inch heatmap
shows the five full instructions and Document, with `pre, 0, 1, ...` on the
audio-encoder axis. Use `--include_short` to show all eleven rows in a taller
figure; all rows are always retained in the CSVs.

Color limits are shared across raw/adjusted plots and all supplied runs with
the same backbone and model size. Qwen and Gemma have separate limits. Pass
all runs being compared in one plotting command to keep their scales aligned.
`--title` in the extraction command overrides the backbone title for a local
variant.
