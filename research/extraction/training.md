# CA1 training and evaluation in Lightning Studio

The accepted [CA1 v2 packet](../../data/research/extraction/sft-pilot-v2/README.md)
contains 320 synthetic training and 64 synthetic validation reports. The
research-only trainer uses the sealed packet, pinned Qwen revision, NF4 QLoRA,
and assistant-only labels. It does not modify the production application.

One interruptible Lightning L4 completed the three-epoch run. Synthetic
validation loss selected `checkpoint-80`; the [training record](../../data/research/extraction/ca1-run-full/run.json)
reports 120 optimizer steps in 4,990 seconds, 14.18 GiB peak allocated, and
validation loss 0.013559 at step 80. Matched synthetic, real-development, and
reserved evaluations are complete. See [results and limitations](evaluations.md).
These instructions document the completed Lightning workflow.

## Prepare the Studio

Use the existing private Studio and its [default Python environment](https://lightning.ai/docs/overview/ai-studio/). The run
used PyTorch 2.8.0+cu128, Transformers 5.16.1, tokenizers 0.23.2, PEFT 0.21.0,
accelerate 1.15.0, and bitsandbytes 0.50.2. The core dependency pins other than PyTorch are in
[requirements-training.txt](requirements-training.txt); W&B is optional and
installed separately below.
The accepted packet loader rejects changed packet files, prompt, schema,
tokenizer, generator source, split membership, or assistant labels. Do not
edit the frozen packet or its source files.

Upload the single private [Studio input archive](../../data/research/extraction/ca1-lightning-input-v1.tar.gz)
to `/teamspace/studios/this_studio`. Its SHA-256 is
`03aa28416620d0bf942749bcb81d6eb1eb2f5ec279ae9d31817f4087fb825ed9`.
It contains the sealed synthetic packet, pinned tokenizer, research code, and
the 13 development and 11 reserved report inputs and targets. The real-report
index contains only IDs and roles; the archive contains no model weights,
adapter, or credentials. Keep it in a private Studio. On the Studio's CPU
machine:

```sh
cd /teamspace/studios/this_studio
sha256sum ca1-lightning-input-v1.tar.gz
tar -xzf ca1-lightning-input-v1.tar.gz
cd ca1
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
python3 -m pip install -r research/extraction/requirements-training.txt
python3 -m research.extraction.train_ca1 check
```

Confirm the archive digest and the check output: **320 train, 64 validation,
maximum 4,742 tokens**. Lightning Studio already provides an environment;
creating another `venv` is unnecessary and was rejected in this Studio.
Download the pinned base weights while on CPU if they are not cached:

```sh
export HF_HOME=/teamspace/studios/this_studio/hf-cache
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554")'
```

## Run on the L4

Switch the Studio to one interruptible L4 and check that CUDA sees it. The
preflight is one forward/backward pass on the longest accepted example; it
does not change model weights.

```sh
cd /teamspace/studios/this_studio/ca1
export HF_HOME=/teamspace/studios/this_studio/hf-cache
nvidia-smi
python3 -m research.extraction.train_ca1 preflight
```

The completed preflight took 23.7 seconds and allocated 16.33 GiB of the
22.03 GiB available to CUDA. For the recorded full run, install W&B 0.30.0
in the same Studio environment, set `WANDB_API_KEY` as a Studio secret, and
start with a new output directory. The key must not appear in shell history,
logs, or archives.

```sh
python3 -m pip install wandb==0.30.0
export WANDB_PROJECT=ca1-pilot-v2
export WANDB_NAME=ca1-pilot-v2-full
unset WANDB_RUN_ID WANDB_RESUME
python3 -m research.extraction.train_ca1 train \
  --output /teamspace/studios/this_studio/ca1-run-full \
  --report-to wandb
```

The run evaluates synthetic validation loss at steps 40, 80, and 120 and
saves checkpoints every 10 optimizer steps. `run.json` records the recipe,
hardware, package versions, and best checkpoint. The output directory must be
empty for a new run. To continue an interrupted run, restore the same L4,
package versions, `HF_HOME`, and W&B settings, then use its latest complete
checkpoint under the same output directory:

```sh
python3 -m research.extraction.train_ca1 train \
  --output /teamspace/studios/this_studio/ca1-run-full \
  --resume /teamspace/studios/this_studio/ca1-run-full/checkpoint-N \
  --report-to wandb
```

Replace `checkpoint-N` with a complete saved checkpoint. [Sleeping the Studio](https://lightning.ai/docs/overview/ai-studio/)
stops the script; it does not pause Python for automatic continuation. Keep
`run.json` with the adapter, optimizer, scheduler, and RNG files if resuming
training. This runner checks the GPU identity and dependency versions, so it
will not resume this L4 run on a different GPU. The completed run's
[selected adapter archive](../../data/research/extraction/ca1-best-adapter.tar.gz)
contains only `run.json`, `checkpoint-80/adapter_config.json`, and
`checkpoint-80/adapter_model.safetensors`; it is sufficient for inference,
not optimizer-state resumption. Its SHA-256 is
`e880cf8637d8df52f80a0073f0922ea176fd4d4fa5db56bef85ad621893f2344`.

## Matched evaluation

The evaluator uses the same base model, tokenizer, prompt, precision, and
greedy decoding for base and adapted outputs. Every report is saved before
moving to the next, so rerun an interrupted command with `--resume`. The
selected adapter must be under its completed run directory. The default
command evaluates only the 64 synthetic validation reports:

```sh
cd /teamspace/studios/this_studio/ca1
export HF_HOME=/teamspace/studios/this_studio/hf-cache
python3 -m research.extraction.evaluate_ca1_adapter \
  --adapter /teamspace/studios/this_studio/ca1-run-full/checkpoint-80 \
  --output /teamspace/studios/this_studio/ca1-eval-full \
  --device cuda
```

The same extracted archive already contains the approved prompt, schema,
minimal role index, and all 24 real-report inputs and targets under
`data/research/extraction/sft-correction-v1/`. The trainer reads only the sealed
synthetic packet. The evaluator selects each real role explicitly; it uses
the same code, prompt, schema, and adapter for both. On the L4, run the two
roles separately:

```sh
cd /teamspace/studios/this_studio/ca1
export HF_HOME=/teamspace/studios/this_studio/hf-cache
python3 -m research.extraction.evaluate_ca1_adapter \
  --adapter /teamspace/studios/this_studio/ca1-run-full/checkpoint-80 \
  --output /teamspace/studios/this_studio/ca1-eval-development-v1 \
  --device cuda --development-only
python3 -m research.extraction.evaluate_ca1_adapter \
  --adapter /teamspace/studios/this_studio/ca1-run-full/checkpoint-80 \
  --output /teamspace/studios/this_studio/ca1-eval-reserved-v1 \
  --device cuda --reserved-only
```

The reserved evaluator uses 8,192 total tokens for ten reports and 16,384
for D0018, recorded separately as `real-reserved-long`. D0018 completed on
the L4 and reached EOS. The user downloaded [synthetic](../../data/research/extraction/ca1-eval-full.tar.gz),
[development](../../data/research/extraction/ca1-eval-development-v1.tar.gz),
and [reserved](../../data/research/extraction/ca1-eval-reserved-v1.tar.gz)
outputs; their digests, scores, and limitations are in [evaluations.md](evaluations.md).
The real reports were not training or checkpoint-selection records. The
reserved set is this checkpoint's historical final test; any later model or
prompt change needs new held-out reports for a fresh final assessment.

To download an output folder from the Studio file menu, make one archive file
in the Studio root, then download that file. For example:

```sh
cd /teamspace/studios/this_studio
tar -czf ca1-eval-development-v1.tar.gz ca1-eval-development-v1
sha256sum ca1-eval-development-v1.tar.gz
```

## Other compute options

| Platform | Opportunity and observed limitation |
| --- | --- |
| [Kaggle](https://www.kaggle.com/docs/notebooks) | Free GPU access may suit smaller experiments; this packet's 4,742-token backward pass ran out of memory on the tested T4. |
| [Yandex Cloud](https://yandex.cloud/en/docs/compute/concepts/gpus) | Paid T4i/V100/A100 VM options permit a separate CUDA run after quota, price, and full-length preflight checks. The current trainer cannot resume the Lightning L4 run on a different GPU. |
| [Colab](https://research.google.com/colaboratory/faq.html) | Easy notebook setup; GPU availability, session length, and persistent storage need checking for the actual account. |
| Local Mac | The evaluator has an MPS path, but full model generation and training were not validated there. |

The Lightning L4 is the only platform on which this exact training and matched
evaluation workflow completed. The merged input archive was assembled after
those runs; the original separate input archives remain historical provenance.
Historical pricing and account limits can change; consult each provider before
starting another run.
