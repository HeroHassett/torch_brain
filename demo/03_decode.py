"""Act 4 · Decoding: the pretrained POYO+ reads held-out spikes and predicts movement.

Every test trial of every recording is cut into overlapping 1 s windows, decoded,
stitched back together, and scored with R² (1.0 = perfect, 0.0 = predicting the mean).

    python demo/03_decode.py                       # pretrained model, 12 recordings
    python demo/03_decode.py --monkey-t            # the finetuned model on monkey T
    python demo/03_decode.py --ckpt path/to.ckpt --session <session_id>
"""

import argparse
from pathlib import Path

import common  # noqa: F401
from common import (
    C,
    DATA_ROOT,
    FIG_DIR,
    FINETUNED_CKPT,
    MONKEYS,
    PRETRAINED_CKPT,
    REPO,
    Timer,
    banner,
    console,
    device_label,
    load_model,
    pick_device,
    rel,
    score_bar,
    score_style,
    section,
    show_figure,
)

import numpy as np
import torch
import torchmetrics
from rich import box
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from torch_brain.data import Dataset, collate
from torch_brain.data.dataset import DatasetIndex
from torch_brain.models import POYOPlusTokenizer
from torch_brain.registry import MODALITIY_REGISTRY
from torch_brain.utils.stitcher import stitch

READOUT = "cursor_velocity_2d"
TARGET_COLORS = [C["red"], C["orange"], C["yellow"], C["green"], C["cyan"], C["blue"], C["magenta"], "#f5bde6"]


def windows_for(interval, length=1.0, step=0.5):
    """Same windowing as the evaluation sampler: overlapping windows + one flush right."""
    out = []
    for s, e in zip(interval.start, interval.end):
        if e - s < length:
            continue
        starts = list(np.arange(s, e - length + 1e-9, step))
        if starts[-1] + length < e:
            starts.append(e - length)
        out.extend(float(x) for x in starts)
    return out


@torch.no_grad()
def decode_recording(model, ds, recording_id, device, batch_size=64, progress=None, task=None):
    starts = windows_for(ds.get_sampling_intervals()[recording_id])
    preds, targets, times, weights = [], [], [], []
    for b in range(0, len(starts), batch_size):
        chunk = starts[b : b + batch_size]
        batch = collate([ds[DatasetIndex(recording_id, s, s + 1.0)] for s in chunk])
        inputs = {
            k: batch[k].to(device)
            for k in [
                "input_unit_index", "input_timestamps", "input_token_type", "input_mask",
                "latent_index", "latent_timestamps",
                "output_session_index", "output_timestamps", "output_decoder_index",
            ]
        }
        out = model(**inputs, unpack_output=True)
        mask = batch["output_decoder_index"] == MODALITIY_REGISTRY[READOUT].id
        counts = mask.sum(1).tolist()
        tgt = torch.split(batch["target_values"][READOUT], counts)
        wgt = torch.split(batch["target_weights"][READOUT], counts)
        for i in range(len(chunk)):
            if counts[i] == 0:
                continue
            preds.append(out[i][READOUT].float().cpu())
            targets.append(tgt[i].float())
            weights.append(wgt[i].float())
            ts = batch["output_timestamps"][i][mask[i]].double() + float(batch["absolute_start"][i])
            times.append(ts)
        if progress is not None:
            progress.advance(task, len(chunk))

    pred, target, ts = torch.cat(preds), torch.cat(targets), torch.cat(times)
    w = torch.cat(weights)
    ts = torch.round(ts * 1e4) / 1e4
    uniq = torch.unique(ts, sorted=True)
    pred, target = stitch(ts, pred), stitch(ts, target)
    # like the training-time metric: ignore annotated cursor-tracking artifacts
    keep = stitch(ts, w.unsqueeze(-1)).squeeze(-1) > 0
    r2 = torchmetrics.functional.r2_score(pred[keep], target[keep]).item()
    return dict(r2=r2, pred=pred.numpy(), target=target.numpy(), t=uniq.numpy(), n_windows=len(starts))


def results_table(results, ds):
    table = Table(
        title="R² on held-out test trials (never seen during training)",
        title_style="title",
        box=box.ROUNDED,
        border_style=C["grey"],
        header_style="accent",
        show_footer=True,
        footer_style="bold",
    )
    table.add_column("recording", footer="average")
    table.add_column("subject")
    table.add_column("task")
    table.add_column("neurons", justify="right")
    table.add_column("windows", justify="right", style=C["grey"])
    table.add_column("R²", justify="right")
    table.add_column("", min_width=24)
    for rid, res in sorted(results.items(), key=lambda kv: -kv[1]["r2"]):
        session = rid.split("/")[-1]
        name, color = MONKEYS.get(session[0], (session[0], C["fg"]))
        n_units = len(ds._data_objects[rid].units.id)
        table.add_row(
            session,
            Text.from_markup(f"[bold {color}]●[/] {name}"),
            "center-out" if "center_out" in session else "random-target",
            str(n_units),
            str(res["n_windows"]),
            Text(f"{res['r2']:.3f}", style=score_style(res["r2"])),
            score_bar(res["r2"]),
        )
    avg = float(np.mean([r["r2"] for r in results.values()]))
    table.columns[5].footer = Text(f"{avg:.3f}", style=score_style(avg))
    table.columns[6].footer = score_bar(avg)
    return table, avg


def terminal_trace(res, std, title):
    """Decoded vs true velocity, drawn with braille characters right in the terminal."""
    import plotext as plt

    t, pred, target = res["t"], res["pred"] * std, res["target"] * std
    # test trials are separate intervals: put the first few back to back
    gaps = np.where(np.diff(t) > 0.05)[0]
    bounds = [0] + list(gaps + 1) + [len(t)]
    xs, ys_true, ys_pred, offset = [], [], [], 0.0
    for k in range(min(4, len(bounds) - 1)):
        sl = slice(bounds[k], bounds[k + 1])
        x = t[sl] - t[sl][0] + offset
        xs.extend(x.tolist())
        ys_true.extend(target[sl, 0].tolist())
        ys_pred.extend(pred[sl, 0].tolist())
        offset = x[-1] + 0.25
    plt.clf()
    plt.theme("clear")  # transparent: inherits the terminal's own colour scheme
    plt.plotsize(min(console.width - 4, 140), 18)
    plt.plot(xs, ys_true, label="true vₓ", color=(166, 218, 149), marker="braille")
    plt.plot(xs, ys_pred, label="POYO+ vₓ", color=(198, 160, 246), marker="braille")
    plt.title(title)
    plt.xlabel("time (s) · first 4 held-out trials, back to back")
    plt.ylabel("cm/s")
    console.print(Text.from_ansi(plt.build()))


def make_figure(res, data, std, session, path, r2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg, grid = "#24273a", "#cad3f5", "#363a4f"
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "axes.edgecolor": grid,
        "axes.labelcolor": fg, "text.color": fg, "xtick.color": fg, "ytick.color": fg, "font.size": 11,
    })
    t, pred, target = res["t"], res["pred"] * std, res["target"] * std
    has_reaches = hasattr(data, "movement_phases") and hasattr(data.movement_phases, "reach_period")

    fig = plt.figure(figsize=(15, 6.8), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], wspace=0.12, hspace=0.25)
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_f = fig.add_subplot(gs[:, 1])

    # concatenate the first ~8 test trials on a compressed time axis
    gaps = np.where(np.diff(t) > 0.05)[0]
    n_trials = min(8, len(gaps) + 1)
    bounds = [0] + list(gaps[: n_trials - 1] + 1) + [gaps[n_trials - 1] + 1 if len(gaps) >= n_trials else len(t)]
    offset = 0.0
    for k in range(n_trials):
        sl = slice(bounds[k], bounds[k + 1])
        x = t[sl] - t[sl][0] + offset
        for ax, dim in ((ax_x, 0), (ax_y, 1)):
            ax.plot(x, target[sl, dim], color=C["green"], lw=2.0, alpha=0.9, label="true" if k == 0 else None)
            ax.plot(x, pred[sl, dim], color=C["magenta"], lw=1.6, label="POYO+" if k == 0 else None)
        offset = x[-1] + 0.4
        if k < n_trials - 1:
            for ax in (ax_x, ax_y):
                ax.axvline(offset - 0.2, color=grid, lw=1, ls=":")
    ax_x.set_ylabel("vₓ (cm/s)")
    ax_y.set_ylabel("v_y (cm/s)")
    ax_y.set_xlabel("time (s) · first held-out test trials, back to back")
    ax_x.set_title(f"{session}  ·  R² = {r2:.3f}", loc="left", color=C["blue"], fontsize=12)
    ax_x.legend(loc="upper right", frameon=False, ncol=2)

    if has_reaches:
        rp, tr = data.movement_phases.reach_period, data.trials
        pos, ct = data.cursor.pos, data.cursor.timestamps
        drawn = 0
        for s, e in zip(rp.start, rp.end):
            m = (t >= s) & (t <= e)
            if m.sum() < 10:
                continue
            k = np.searchsorted(tr.start, s, side="right") - 1
            tid = tr.target_id[k]
            if np.isnan(tid):
                continue
            col = TARGET_COLORS[int(tid)]
            p0 = pos[np.searchsorted(ct, t[m][0])]
            dt = np.gradient(t[m])[:, None]
            true_xy = p0 + np.cumsum(target[m] * dt, axis=0)
            pred_xy = p0 + np.cumsum(pred[m] * dt, axis=0)
            ax_f.plot(true_xy[:, 0], true_xy[:, 1], color=col, lw=1.0, alpha=0.35, ls="--")
            ax_f.plot(pred_xy[:, 0], pred_xy[:, 1], color=col, lw=2.0, alpha=0.95)
            drawn += 1
        ax_f.set_title(f"decoded reaches (solid) vs real (dashed) · {drawn} test reaches", color=C["blue"], fontsize=11)
        ax_f.set_aspect("equal")
    else:
        ax_f.scatter(target[:, 0], pred[:, 0], s=2, alpha=0.3, color=C["cyan"], label="vₓ")
        ax_f.scatter(target[:, 1], pred[:, 1], s=2, alpha=0.3, color=C["yellow"], label="v_y")
        lim = np.percentile(np.abs(target), 99)
        ax_f.plot([-lim, lim], [-lim, lim], color=fg, lw=1, ls=":")
        ax_f.set_xlim(-lim, lim)
        ax_f.set_ylim(-lim, lim)
        ax_f.set_xlabel("true")
        ax_f.set_ylabel("decoded")
        ax_f.legend(frameon=False)
        ax_f.set_title("decoded vs true velocity", color=C["blue"], fontsize=11)
    for ax in (ax_x, ax_y, ax_f):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax_f.set_xticks([]) if has_reaches else None
    ax_f.set_yticks([]) if has_reaches else None
    fig.savefig(path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="checkpoint (default: demo pretrained model)")
    parser.add_argument("--monkey-t", action="store_true", help="evaluate the model finetuned on monkey T")
    parser.add_argument("--session", default=None, help="session to plot (default: best center-out one)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-terminal-plot", action="store_true")
    parser.add_argument("--json", default=None, help="also write the results to this JSON file")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve() if args.ckpt else (FINETUNED_CKPT if args.monkey_t else PRETRAINED_CKPT)
    device = pick_device(args.device)
    act = "ACT 7 · A MONKEY IT HAS NEVER SEEN" if args.monkey_t else "ACT 4 · READING MINDS (WELL, MOTOR CORTEX)"
    banner("held-out trials · overlapping windows stitched together · R² per recording", act=act)

    with console.status("[title]loading checkpoint…", spinner="dots12", spinner_style=C["magenta"]):
        model, cfg, ckpt = load_model(ckpt_path, device)
        tokenizer = POYOPlusTokenizer(
            model.unit_emb.tokenizer, model.session_emb.tokenizer,
            decoder_registry=MODALITIY_REGISTRY, latent_step=1 / 8,
            num_latents_per_step=cfg.model.num_latents, eval=True,
        )
        ds = Dataset(root=str(DATA_ROOT), config=cfg.dataset, split="test", transform=tokenizer)
        ds.disable_data_leakage_check()
    std = np.array(cfg.dataset[0]["config"]["multitask_readout"][0].get("normalize_std", [1.0, 1.0]))
    console.print(
        Text.assemble(
            ("✔ ", "good"), (rel(ckpt_path), f"underline {C['cyan']}"),
            ("  ·  ", C["grey"]), (f"{len(ds.recording_dict)} recordings", C["yellow"]),
            ("  ·  ", C["grey"]), (device_label(device), "accent"),
        )
    )

    section("Decoding every held-out trial", "🧠")
    rec_ids = list(ds.recording_dict.keys())
    total = sum(len(windows_for(ds.get_sampling_intervals()[r])) for r in rec_ids)
    results = {}
    with Timer() as timer, Progress(
        SpinnerColumn("dots12", style=C["magenta"]),
        TextColumn("[title]{task.fields[name]:<34}"),
        BarColumn(bar_width=None, complete_style=C["magenta"], finished_style=C["green"]),
        MofNCompleteColumn(),
        TextColumn("[muted]windows"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("decode", total=total, name="")
        for rid in rec_ids:
            progress.update(task, name=rid.split("/")[-1])
            results[rid] = decode_recording(model, ds, rid, device, progress=progress, task=task)
    seconds_decoded = sum(len(r["t"]) for r in results.values()) / 100
    elapsed = timer.elapsed
    table, avg = results_table(results, ds)
    console.print(table)
    console.print(
        Text.assemble(
            ("   decoded ", C["grey"]), (f"{seconds_decoded / 60:.1f} minutes", C["yellow"]),
            (" of neural activity in ", C["grey"]), (f"{timer.elapsed:.1f} s", C["green"]),
            (" (incl. data loading)", C["grey"]),
        )
    )

    # featured session
    if args.session:
        featured = next(r for r in rec_ids if args.session in r)
    else:
        co = [r for r in rec_ids if "center_out" in r] or rec_ids
        featured = max(co, key=lambda r: results[r]["r2"])
    session = featured.split("/")[-1]
    res = results[featured]

    if not args.no_terminal_plot:
        section(f"Right in the terminal · {session}", "📈")
        terminal_trace(res, std, f"horizontal cursor velocity · true vs decoded · R² {res['r2']:.3f}")

    section("Decoded movements", "🎯")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / ("05_monkey_t_decoding.png" if args.monkey_t else "03_decoding.png")
    with console.status("[title]rendering figure…", spinner="dots12", spinner_style=C["magenta"]):
        make_figure(res, ds._data_objects[featured], std, session, path, res["r2"])
    show_figure(path, "velocity traces (left) · reach trajectories rebuilt from decoded velocity (right)")

    if args.json:
        import json

        Path(args.json).write_text(json.dumps({
            "average_r2": avg,
            "r2": {rid.split("/")[-1]: r["r2"] for rid, r in results.items()},
            "units": {rid.split("/")[-1]: int(len(ds._data_objects[rid].units.id)) for rid in results},
            "windows": int(sum(r["n_windows"] for r in results.values())),
            "minutes_decoded": seconds_decoded / 60,
            "seconds_elapsed": elapsed,
            "featured": session,
        }, indent=2))

    console.print(
        Panel(
            Text.assemble(
                ("Average R² ", "accent"), (f"{avg:.3f}", score_style(avg)),
                (f" across {len(results)} recordings", C["fg"]),
                (" · decoded from raw spikes, on held-out trials, on a laptop GPU.", C["grey"]),
            ),
            border_style=C["green"],
        )
    )


if __name__ == "__main__":
    main()
