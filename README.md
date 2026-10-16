# D-PAL training code

This anonymous supplement contains the DFlash training code for D-PAL. It is
based on [SpecForge](https://github.com/sgl-project/SpecForge), and the
upstream MIT license is kept in `LICENSE`. It contains code and example
configs, not model weights or training data.

| Paper objective | `training.loss_type` | Local loss |
| --- | --- | --- |
| D-PAL[LA] | `dynamic-nla` | log acceptance, −log(1 − TV(p, q)) |
| D-PAL[KL] | `dynamic-kl` | forward KL(p ‖ q) |
| D-PAL[Rényi] (default) | `dpard` | Rényi divergence of order 1/2, −2 log Σ_v √(p_v q_v) |

The `loss_type` values are internal identifiers. All three objectives use the
same detached position weights, w_t = Σ_{k≥t} Π_{i≤k} s_i with
s_i = ρ + (1 − ρ)(1 − TV(p_i, q_i)). The code is in
`specforge/algorithms/common/dflash_family_model.py`: `_unary_renyi_overlap`
computes the local losses and `_dpace_weight` the weights.

The smoothing floor ρ is read from `training.dpard_alpha` for `dpard` and from
`training.dpace_alpha` for `dynamic-kl` and `dynamic-nla`. The recipe sets both
to 0.3; set both when changing ρ.

For these objectives, the frozen target head projects each sequence position
once per forward pass. Objective chunks gather their positions from the shared
logits, and activation-checkpoint replay does not repeat the projection. The
projection is not reused across batches or epochs, and the loss formulas and
the `sequence_anchor_mean` reduction are unchanged. Detailed diagnostics keep
their own projection path, so the recipe sets a large `log_interval`.

## Setup

On a CUDA machine, install from the repository root:

```bash
pip install -e .
```

The pinned versions (torch 2.13.0, transformers 5.12.1, SGLang 0.5.18) are used
for feature capture and training. The paper measures acceptance length and
speedup with the DFlash backend of SGLang 0.5.19.

Run all commands from the repository root; config paths are resolved relative
to the current directory.

## One-step example

This example checks that the pipeline runs end to end on 16 ShareGPT
conversations. It is not the paper's training run.

```bash
python scripts/prepare_data.py --dataset sharegpt --sample-size 16

torchrun --nproc_per_node=1 scripts/prepare_hidden_states.py \
  --strategy dflash \
  --target-model-path Qwen/Qwen3-4B \
  --draft-model-config configs/qwen3-4b-dpal-b16.json \
  --data-path cache/dataset/sharegpt_train.jsonl \
  --output-path cache/hidden_states/qwen3-4b-dpal \
  --chat-template qwen --max-length 512 \
  --tp-size 1 --batch-size 1 --num-samples 16
```

Train one step with any of the three objectives. The CLI accepts dotted
overrides after the config path; `training.log_interval=1` logs the metrics of
that step.

```bash
specforge train -c examples/configs/offline/colocated/qwen3-4b-dpal.yaml \
  training.loss_type=dpard training.max_steps=1 training.batch_size=1 \
  training.num_anchors=8 data.max_length=512 training.log_interval=1 \
  run_id=example-renyi output_dir=./outputs/example-renyi

specforge train -c examples/configs/offline/colocated/qwen3-4b-dpal.yaml \
  training.loss_type=dynamic-kl training.max_steps=1 training.batch_size=1 \
  training.num_anchors=8 data.max_length=512 training.log_interval=1 \
  run_id=example-kl output_dir=./outputs/example-kl

specforge train -c examples/configs/offline/colocated/qwen3-4b-dpal.yaml \
  training.loss_type=dynamic-nla training.max_steps=1 training.batch_size=1 \
  training.num_anchors=8 data.max_length=512 training.log_interval=1 \
  run_id=example-la output_dir=./outputs/example-la
```

## Full run

`examples/configs/offline/colocated/qwen3-4b-dpal.yaml` records the paper's
Qwen3-4B setting: three draft layers, block size 16, 3072-token sequences, 512
anchors per sequence, six epochs, batch size 8 on one GPU, learning rate 6e-4
with cosine decay and 4% warm-up, seed 42, and ρ = 0.3.

1. Generate the training responses with the frozen target at temperature 0.7,
   for example with `scripts/regenerate_train_data.py`, whose default
   temperature is 0.7. Appendix A of the paper describes the training data.
2. Capture features with `--max-length 3072` and without `--num-samples`,
   writing to a new `--output-path`.
3. Train with the recipe, overriding only `data.hidden_states_path` and
   `training.loss_type`. Do not pass the `max_steps`, `batch_size`,
   `num_anchors`, `max_length`, or `log_interval` overrides from the example.

Use new file and directory names rather than overwriting the example's:
`prepare_data.py` skips conversion when its output file already exists, and
the capture cache is keyed by the data path and settings, not by file
contents.

Each cached record holds `input_ids`, `loss_mask`, `hidden_states` from the
three target layers, and `target_last_hidden_states`, the target's final hidden
state. The frozen output head maps it to the target distribution p_t used by
all three losses.

## Tests

Check the local losses, the weights, and the shared target projection without
loading a model:

```bash
python -m unittest discover -s tests/test_modeling -p test_dflash1_six_actors.py
python -m unittest discover -s tests/test_modeling -p test_target_projection_cache.py
```

On a CUDA machine the loss helper is compiled with `torch.compile`, which needs
a C++ compiler.
