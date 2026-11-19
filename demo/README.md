# 🧠 POYO+ · a terminal demo of torch_brain

> **Transformers that read neural spikes.** POYO+ turns every spike from every neuron
> into a token, compresses them with a Perceiver-style encoder, and decodes behaviour,
> here the velocity of a monkey's hand while reaching, from **raw motor-cortex activity**.

## What you are looking at

| | |
|---|---|
| **Model** | POYO+ (`torch_brain.models.POYOPlus`), 5.2 M parameters, 6 latent self-attention layers |
| **Data** | Perich, Miller et al. 2018 · DANDI:000688 · 4 macaques · M1/PMd Utah arrays · 111 sessions |
| **Task** | decode 2-D cursor velocity at 100 Hz from spikes (center-out & random-target reaching) |
| **Hardware** | Apple-Silicon GPU through PyTorch's Metal (MPS) backend, or CPU, or CUDA |

## The acts

| act | command | what happens |
|---|---|---|
| 1 | `python demo/01_explore_data.py` | tour of the brainset, a truecolour spike heatmap, spikes → tokens |
| 2 | `python demo/02_model_tour.py` | architecture tree, a real forward pass, inference speed |
| 3 | `python train.py --config-name demo_live` | train POYO+ from scratch, live, with Rich progress bars |
| 4 | `python demo/03_decode.py` | the pretrained model decodes every held-out trial · R² per session |
| 5 | `python demo/04_live_bci.py` | 🎮 live brain-computer-interface replay, drawn in braille |
| 6 | `python finetune.py --config-name demo_finetune` | transfer to **monkey T**, an animal the model never saw |
| 7 | `python demo/03_decode.py --monkey-t` · `python demo/04_live_bci.py --monkey-t` | decode the new monkey |

Run everything back to back with `zsh demo/showcase.sh`, or check the setup first with
`python demo/check_setup.py`.

## How POYO+ works

```text
 spikes ──► (neuron id, time) tokens ──► unit embedding + rotary time
                                              │  cross-attention
                                              ▼
                               latent tokens (16 every 125 ms)
                                              │  6 × self-attention
                                              ▼
 (session, task, time) queries ──► cross-attention ──► linear head ──► cursor velocity
```

* **No binning, no fixed channel order.** Neurons are *learned embeddings*, so recordings
  with different neurons, arrays, or animals all feed the same network.
* **Transfer by "unit identification".** For a new animal, freeze the network and learn
  only the new neurons' embeddings, then finetune everything.

## Under the hood (changes made for this demo)

* Apple-Silicon support: float32 timestamps (MPS has no float64), a contention-free
  embedding backward pass (**~1.6× faster training steps**), and a sync-free Lamb optimizer.
* Evaluation fix: windows are now stitched into the *right* recording (previously every
  window of a batch was credited to the first sequence of that batch).
* Data hygiene: annotated cursor-tracking glitches (`cursor_outlier_segments`, up to
  1300 cm/s) are ignored by the loss and the metrics (`ignore_intervals` readout option).
  Same compute, **2× the R²** on the live run, and no more NaN blow-ups.
* `train.py` / `finetune.py`: auto device selection (CUDA → MPS → CPU), TensorBoard logging
  when W&B is off, colourful Rich progress bars and R² tables, iTerm2 notifications.

## Credits

* POYO: Azabou et al., *A Unified, Scalable Framework for Neural Population Decoding*, NeurIPS 2023.
* POYO+: Azabou et al., *Multi-session, multi-task neural decoding from distinct cell-types and brain regions*, ICLR 2025.
* Data: Perich, Gallego & Miller, *A neural population mechanism for rapid learning*, Neuron 2018 (DANDI:000688).
