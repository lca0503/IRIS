# ms-swift training patch

IRIS training uses a patched version of [ms-swift](https://github.com/modelscope/ms-swift).
[base_commit.txt](base_commit.txt) specifies the upstream revision, and [iris.patch](iris.patch) contains the changes.
The patched checkout lives in `third_party/ms-swift`, which is ignored by Git.

## Setup

Run from the repository root in your training environment:

```bash
bash scripts/setup_training.sh
```

The script downloads the pinned revision, applies the patch, and installs the training dependencies.
See the [training guide](../../examples/embed/training/README.md) for environment setup and training commands.

## Changes

The patch adds:

- Qwen and Gemma IRIS models from `iris.embed.modeling`.
- Query/document templates and last-valid-token pooling.
- MiniLM tokens and document metadata in training batches.
- Metadata gathering across all ranks and false-negative masking.
- Saving and export of trained audio modules with LLM LoRA.
- Gemma audio SDPA, padding masks, and pooling from raw hidden states.

Inference uses the Transformers model code without ms-swift.

## Update the patch

After editing the checkout, regenerate the patch from the repository root:

```bash
git -C third_party/ms-swift diff --binary > patches/ms-swift/iris.patch
```

Include only changes needed for IRIS training.
Keep the upstream copyright headers and license.
