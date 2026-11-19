"""Act 2 · Inside POYO+: architecture tour + a real forward pass, live on the GPU.

    python demo/02_model_tour.py
"""

import argparse

import common  # noqa: F401
from common import (
    C,
    DATA_ROOT,
    PRETRAINED_CKPT,
    REPO,
    Timer,
    banner,
    console,
    device_label,
    human,
    load_model,
    pick_device,
    section,
)

import numpy as np
import torch
from omegaconf import OmegaConf
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from torch_brain.data import Dataset, collate
from torch_brain.data.dataset import DatasetIndex
from torch_brain.models import POYOPlusTokenizer
from torch_brain.registry import MODALITIY_REGISTRY

DATASET_CFG = REPO / "configs" / "dataset" / "demo_pretrain.yaml"


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def architecture_tree(model, cfg) -> Tree:
    m = cfg.model
    p = lambda mod: f"[muted]{human(n_params(mod))} params[/]"
    root = Tree(
        Text.assemble(("POYO+ ", "title"), (f"· {human(n_params(model))} parameters", "muted")),
        guide_style=C["grey"],
    )
    inp = root.add(f"[accent]① Input tokens[/]  [muted]one per spike[/]")
    inp.add(f"[cyan]unit embedding[/]  {len(model.unit_emb.vocab) - 1} neurons × {m.dim}d  {p(model.unit_emb)}")
    inp.add(f"[cyan]token-type embedding[/]  spike / unit-start / unit-end  {p(model.token_type_emb)}")
    inp.add("[cyan]rotary time embedding[/]  (relative spike timing, no binning)")
    enc = root.add(f"[accent]② Encoder[/]  latents ← cross-attention ← spikes  {p(model.enc_atn)}")
    enc.add(f"[cyan]{m.num_latents} latents every 125 ms[/] → {m.num_latents * 8} latents / second")
    enc.add(f"[cyan]{m.cross_heads} heads[/] × {m.dim_head}d, then feed-forward {p(model.enc_ffn)}")
    proc = root.add(f"[accent]③ Processor[/]  {m.depth} × self-attention blocks on the latents  {p(model.proc_layers)}")
    proc.add(f"[cyan]{m.self_heads} heads[/] × {m.dim_head}d · rotary time · residual dropout {m.lin_dropout}")
    dec = root.add(f"[accent]④ Decoder[/]  output queries → cross-attention → latents  {p(model.dec_atn)}")
    dec.add(f"[cyan]session embedding[/]  {len(model.session_emb.vocab) - 1} recordings  {p(model.session_emb)}")
    dec.add(f"[cyan]task embedding[/]  which behaviour to read out  {p(model.task_emb)}")
    out = root.add(f"[accent]⑤ Multitask readout[/]  one linear head per task  {p(model.readout)}")
    out.add(f"[green]cursor_velocity_2d[/] (used here) + {len(MODALITIY_REGISTRY) - 1} registered tasks "
            "[muted](gratings, natural movies, pupil, running speed, …)[/]")
    return root


def sample_batch(model, n=8, seed=0):
    tokenizer = POYOPlusTokenizer(
        model.unit_emb.tokenizer,
        model.session_emb.tokenizer,
        decoder_registry=MODALITIY_REGISTRY,
        latent_step=1 / 8,
        num_latents_per_step=model.latent_emb.num_embeddings,
        eval=True,
    )
    ds = Dataset(root=str(DATA_ROOT), config=OmegaConf.load(DATASET_CFG), split="test", transform=tokenizer)
    ds.disable_data_leakage_check()
    rng = np.random.default_rng(seed)
    intervals = ds.get_sampling_intervals()
    rec_ids = list(intervals.keys())
    picks = rng.choice(len(rec_ids), size=n, replace=len(rec_ids) < n)
    samples, names = [], []
    for k in picks:
        rid = rec_ids[k]
        iv = intervals[rid]
        lengths = iv.end - iv.start
        j = int(np.argmax(lengths >= 1.0))
        start = float(iv.start[j])
        samples.append(ds[DatasetIndex(rid, start, start + 1.0)])
        names.append(rid.split("/")[-1])
    return collate(samples), names


def flow_table(model, batch, device):
    shapes = {}

    def hook(name):
        def fn(_, args, out):
            shapes[name] = out
        return fn

    handles = [
        model.unit_emb.register_forward_hook(hook("unit_emb")),
        model.enc_atn.register_forward_hook(hook("enc_atn")),
        model.proc_layers[-1][0].register_forward_hook(hook("proc")),
        model.dec_atn.register_forward_hook(hook("dec_atn")),
    ]
    inputs = {k: v for k, v in batch.items() if k not in ("target_values", "target_weights", "session_id", "absolute_start", "output_subtask_index")}
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, unpack_output=False)
    for h in handles:
        h.remove()

    B = batch["input_unit_index"].shape[0]
    n_real = int(batch["input_mask"].sum())
    table = Table(box=box.ROUNDED, border_style=C["grey"], header_style="accent", title="one forward pass, step by step", title_style="title")
    table.add_column("stage")
    table.add_column("tensor shape", style=C["yellow"], justify="center")
    table.add_column("what it is", style=C["fg"])
    fmt = lambda t: " × ".join(str(s) for s in t.shape)
    table.add_row("spike tokens", fmt(batch["input_unit_index"]), f"{B} windows of 1 s, padded · {n_real:,} real spike tokens")
    table.add_row("unit embeddings", fmt(shapes["unit_emb"]), "one learned vector per (neuron, spike)")
    table.add_row("latents after encoder", fmt(shapes["enc_atn"]), "spikes compressed into a fixed-size latent set")
    table.add_row("latents after processor", fmt(shapes["proc"]), "self-attention mixes information over time")
    table.add_row("decoded queries", fmt(shapes["dec_atn"]), "one query per output timestamp (100 Hz)")
    table.add_row("[green]cursor velocity[/]", fmt(out["cursor_velocity_2d"]), "(vₓ, v_y) for every query, all windows chained")
    return table, inputs


def benchmark(model, inputs, device, reps=20):
    def run(dev):
        m = model.to(dev)
        x = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            for _ in range(3):
                m(**x)
            if dev.type == "mps":
                torch.mps.synchronize()
            with Timer() as t:
                for _ in range(reps):
                    out = m(**x)
                    out["cursor_velocity_2d"].sum().item()
        return t.elapsed / reps

    results = [(device_label(device), run(device))]
    if device.type != "cpu":
        results.append(("CPU", run(torch.device("cpu"))))
        model.to(device)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)

    banner("Perceiver-style transformer · rotary time embeddings · multitask readout", act="ACT 2 · INSIDE POYO+")

    with console.status("[title]loading the pretrained POYO+ checkpoint…", spinner="dots12", spinner_style=C["magenta"]):
        model, cfg, ckpt = load_model(PRETRAINED_CKPT, device)
    console.print(
        Text.assemble(
            ("✔ ", "good"), ("checkpoint ", C["fg"]), (str(PRETRAINED_CKPT.relative_to(REPO)), f"underline {C['cyan']}"),
            ("  ·  trained for ", C["grey"]), (f"{ckpt['epoch'] + 1} epochs / {ckpt['global_step']:,} steps", C["yellow"]),
            ("  ·  running on ", C["grey"]), (device_label(device), "accent"),
        )
    )

    section("Architecture", "🏗 ")
    console.print(architecture_tree(model, cfg))

    section("A real batch through the network", "🔬")
    with console.status("[title]slicing 8 random one-second windows from 12 recordings…", spinner="dots12", spinner_style=C["magenta"]):
        batch, names = sample_batch(model)
    console.print(Text("windows from: ", style="muted") + Text(", ".join(sorted(set(names))), style=C["cyan"]))
    table, inputs = flow_table(model, batch, device)
    console.print(table)

    section("How fast is it?", "⏱ ")
    with console.status("[title]benchmarking inference…", spinner="dots12", spinner_style=C["magenta"]):
        results = benchmark(model, inputs, device)
    t = Table(box=box.SIMPLE_HEAVY, header_style="accent", border_style=C["grey"])
    t.add_column("device")
    t.add_column("8 s of neural data", justify="right", style=C["yellow"])
    t.add_column("per 1 s window", justify="right", style=C["green"])
    t.add_column("real-time factor", justify="right")
    for label, sec in results:
        t.add_row(label, f"{sec * 1000:.1f} ms", f"{sec / 8 * 1000:.2f} ms", f"{8 / sec:,.0f}× faster than real time")
    console.print(t)
    console.print(
        Panel(
            Text.assemble(
                ("Takeaway: ", "accent"),
                ("POYO+ reads raw spikes from any set of neurons and decodes behaviour ", C["fg"]),
                ("hundreds of times faster than real time", "good"),
                (" on a laptop.", C["fg"]),
            ),
            border_style=C["green"],
        )
    )


if __name__ == "__main__":
    main()
