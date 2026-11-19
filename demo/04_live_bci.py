"""Act 5 · Live BCI replay: watch POYO+ steer a cursor from spikes, in the terminal.

Held-out center-out trials are replayed in real time. Every 50 ms, POYO+ looks at the
*previous* second of spikes only (causal decoding, like a real brain-computer
interface) and outputs the cursor velocity. Integrating that velocity moves the
magenta cursor; the green one is where the monkey really moved.

    python demo/04_live_bci.py                 # pretrained model, monkey C
    python demo/04_live_bci.py --monkey-t      # finetuned model on never-seen monkey T
    python demo/04_live_bci.py --trials 12 --speed 0.5
"""

import argparse
import time
from pathlib import Path

import common  # noqa: F401
from common import (
    C,
    DATA_ROOT,
    FINETUNED_CKPT,
    PRETRAINED_CKPT,
    banner,
    console,
    device_label,
    load_model,
    pick_device,
    rel,
    score_style,
)

import numpy as np
import torch
from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from torch_brain.data import Dataset, collate
from torch_brain.data.dataset import DatasetIndex
from torch_brain.models import POYOPlusTokenizer
from torch_brain.registry import MODALITIY_REGISTRY

READOUT = "cursor_velocity_2d"
STEP = 0.05  # decode every 50 ms
TARGET_COLORS = [C["red"], C["orange"], C["yellow"], C["green"], C["cyan"], C["blue"], C["magenta"], "#f5bde6"]
ARROWS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]


# ---------------------------------------------------------------- braille canvas
_DOT = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]


class Canvas:
    """A tiny braille canvas: each character cell holds 2×4 dots → smooth lines."""

    def __init__(self, width, height, extent, center=(0.0, 0.0)):
        self.w, self.h, self.extent = width, height, extent
        self.cx, self.cy = center
        self.cells = {}
        self.overlay = {}

    def _to_dots(self, x, y):
        x, y = x - self.cx, y - self.cy
        sx = (x + self.extent) / (2 * self.extent) * (2 * self.w - 1)
        sy = (self.extent - y) / (2 * self.extent) * (4 * self.h - 1)
        return int(round(sx)), int(round(sy))

    def dot(self, px, py, color):
        if 0 <= px < 2 * self.w and 0 <= py < 4 * self.h:
            key = (px // 2, py // 4)
            bits, _ = self.cells.get(key, (0, color))
            self.cells[key] = (bits | _DOT[py % 4][px % 2], color)

    def line(self, p0, p1, color):
        (x0, y0), (x1, y1) = self._to_dots(*p0), self._to_dots(*p1)
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(n + 1):
            self.dot(round(x0 + (x1 - x0) * i / n), round(y0 + (y1 - y0) * i / n), color)

    def polyline(self, pts, color):
        for a, b in zip(pts[:-1], pts[1:]):
            self.line(a, b, color)

    def circle(self, cx, cy, r, color, n=24):
        pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, n + 1)]
        self.polyline(pts, color)

    def mark(self, x, y, char, color):
        px, py = self._to_dots(x, y)
        self.overlay[(px // 2, py // 4)] = (char, color)

    def render(self) -> Text:
        out = Text()
        for cy in range(self.h):
            for cx in range(self.w):
                if (cx, cy) in self.overlay:
                    ch, col = self.overlay[(cx, cy)]
                    out.append(ch, style=f"bold {col}")
                elif (cx, cy) in self.cells:
                    bits, col = self.cells[(cx, cy)]
                    out.append(chr(0x2800 + bits), style=col)
                else:
                    out.append(" ")
            if cy < self.h - 1:
                out.append("\n")
        return out


# ---------------------------------------------------------------- decoding
def pick_trials(data, n_trials):
    """Held-out (test) trials of a center-out session that contain a reach."""
    te, rp, tr = data.test_domain, data.movement_phases.reach_period, data.trials
    trials = []
    for s, e in zip(te.start, te.end):
        reaches = [(a, b) for a, b in zip(rp.start, rp.end) if a >= s + 1.0 and b <= e]
        if not reaches:
            continue
        k = np.searchsorted(tr.start, reaches[0][0], side="right") - 1
        tid = tr.target_id[k]
        if np.isnan(tid):
            continue
        trials.append(dict(start=float(s), end=float(e), reach=reaches[0], target=int(tid),
                           angle=float(tr.target_dir[k])))
    # one trial per target direction first, for a nice variety
    seen, ordered = set(), []
    for t in trials:
        if t["target"] not in seen:
            ordered.append(t)
            seen.add(t["target"])
    ordered += [t for t in trials if t not in ordered]
    return ordered[:n_trials]


@torch.no_grad()
def causal_decode(model, ds, rid, trial, device, std):
    """For t = start+1s ... end, decode window [t-1, t] and keep its last 50 ms."""
    ends = np.arange(trial["start"] + 1.0, trial["end"] + 1e-9, STEP)
    times, vels = [], []
    for b in range(0, len(ends), 64):
        chunk = ends[b : b + 64]
        batch = collate([ds[DatasetIndex(rid, float(t) - 1.0, float(t))] for t in chunk])
        keys = ["input_unit_index", "input_timestamps", "input_token_type", "input_mask",
                "latent_index", "latent_timestamps", "output_session_index",
                "output_timestamps", "output_decoder_index"]
        out = model(**{k: batch[k].to(device) for k in keys}, unpack_output=True)
        mask = batch["output_decoder_index"] == MODALITIY_REGISTRY[READOUT].id
        for i, t_end in enumerate(chunk):
            ts = batch["output_timestamps"][i][mask[i]].double().numpy()
            keep = ts > 1.0 - STEP - 1e-6
            times.append(ts[keep] + t_end - 1.0)
            vels.append(out[i][READOUT].float().cpu().numpy()[keep] * std)
    t = np.concatenate(times)
    v = np.concatenate(vels)
    order = np.argsort(t)
    return t[order], v[order]


# ---------------------------------------------------------------- dashboard
STRIP_UNITS = 24  # adjusted to the terminal height in main()


def activity_strip(data, t_now, n_units=None, seconds=1.5, bins=30):
    n_units = n_units or STRIP_UNITS
    ts, ui = data.spikes.timestamps, data.spikes.unit_index
    lo = np.searchsorted(ts, t_now - seconds)
    hi = np.searchsorted(ts, t_now)
    ts, ui = ts[lo:hi], ui[lo:hi]
    if not hasattr(activity_strip, "top"):
        counts = np.bincount(data.spikes.unit_index, minlength=len(data.units.id))
        activity_strip.top = np.argsort(-counts)[:n_units]
    grid = np.zeros((n_units, bins))
    for r, u in enumerate(activity_strip.top):
        h, _ = np.histogram(ts[ui == u], bins=bins, range=(t_now - seconds, t_now))
        grid[r] = h
    peak = max(grid.max(), 2)
    shades = [C["grey"], "#7c6aa6", C["magenta"], "#f5bde6", C["yellow"]]
    text = Text()
    for r in range(n_units):
        for x in grid[r] / peak:
            idx = min(int(x * (len(shades) - 1) + 0.5), len(shades) - 1)
            text.append("▮" if x > 0 else "·", style=shades[idx] if x > 0 else "#363a4f")
        if r < n_units - 1:
            text.append("\n")
    rate = len(ts[ts > t_now - 0.1]) * 10
    return text, rate


def r2(true, pred):
    if len(true) < 5:
        return float("nan")
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-9)))


def frame(canvas_text, data, t_now, trial, k, n, decoded_v, true_v, rate, session, label, history):
    stats = Table.grid(padding=(0, 1))
    stats.add_column(style=C["grey"], justify="right")
    stats.add_column()
    tgt = trial["target"]
    arrow = ARROWS[int(round(np.degrees(trial["angle"]) / 45.0)) % 8]
    stats.add_row("session", Text(session, style=C["cyan"]))
    stats.add_row("trial", Text(f"{k + 1} / {n}", style=C["yellow"]))
    stats.add_row("target", Text(f"{arrow}  {np.degrees(trial['angle']):+.0f}°", style=f"bold {TARGET_COLORS[tgt]}"))
    stats.add_row("time", Text(f"{t_now - trial['start']:5.2f} s", style=C["fg"]))
    in_reach = trial["reach"][0] <= t_now <= trial["reach"][1]
    stats.add_row("phase", Text("REACH" if in_reach else "hold", style="bold " + (C["green"] if in_reach else C["grey"])))
    stats.add_row("spikes", Text(f"{rate:5.0f} /s", style=C["magenta"]))
    sp_true = np.linalg.norm(true_v)
    sp_dec = np.linalg.norm(decoded_v)
    stats.add_row("true speed", Text(f"{sp_true:5.1f} cm/s", style=C["green"]))
    stats.add_row("POYO+ speed", Text(f"{sp_dec:5.1f} cm/s", style=C["magenta"]))
    if history:
        val = r2(np.array(history["true"]), np.array(history["pred"]))
        if not np.isnan(val):
            stats.add_row("running R²", Text(f"{val:.3f}", style=score_style(val)))

    strip, _ = activity_strip(data, t_now)
    right = Group(
        Panel(stats, title="[title]live decoder[/]", border_style=C["grey"], box=box.ROUNDED),
        Panel(strip, title="[title]motor cortex · last 1.5 s[/]", subtitle=f"[muted]{STRIP_UNITS} most active neurons[/]",
              border_style=C["grey"], box=box.ROUNDED),
    )
    legend = Text.assemble(("● ", f"bold {C['green']}"), ("real hand   ", C["fg"]), ("● ", f"bold {C['magenta']}"),
                           ("POYO+ (decoded from spikes)   ", C["fg"]), ("○ ", C["grey"]), ("targets", C["fg"]))
    left = Panel(Group(canvas_text, Text(), legend), title=f"[title]🧠 → 🖱  {label}[/]",
                 border_style=C["magenta"], box=box.HEAVY)
    grid = Table.grid(expand=False, padding=(0, 1))
    grid.add_column()
    grid.add_column()
    grid.add_row(left, right)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey-t", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", default=None, help="also write the causal R² to this JSON file")
    args = parser.parse_args()

    ckpt = Path(args.ckpt).resolve() if args.ckpt else (FINETUNED_CKPT if args.monkey_t else PRETRAINED_CKPT)
    device = pick_device(args.device)
    banner("causal decoding · 50 ms updates · held-out trials replayed in real time",
           act="ACT 7 · LIVE BCI · NEVER-SEEN MONKEY" if args.monkey_t else "ACT 5 · LIVE BRAIN-COMPUTER INTERFACE")

    with console.status("[title]loading model + recording…", spinner="dots12", spinner_style=C["magenta"]):
        model, cfg, _ = load_model(ckpt, device)
        tok = POYOPlusTokenizer(model.unit_emb.tokenizer, model.session_emb.tokenizer,
                                decoder_registry=MODALITIY_REGISTRY, latent_step=1 / 8,
                                num_latents_per_step=cfg.model.num_latents, eval=True)
        ds = Dataset(root=str(DATA_ROOT), config=cfg.dataset, split="test", transform=tok)
        ds.disable_data_leakage_check()
        rec_ids = [r for r in ds.recording_dict if "center_out" in r]
        if args.session:
            rid = next(r for r in rec_ids if args.session in r)
        else:
            default = "t_20130819" if args.monkey_t else "c_20150629"
            rid = next((r for r in rec_ids if default in r), rec_ids[0])
        data = ds._data_objects[rid]
        std = np.array(cfg.dataset[0]["config"]["multitask_readout"][0].get("normalize_std", [1.0, 1.0]))
        trials = pick_trials(data, args.trials)
    session = rid.split("/")[-1]
    console.print(Text.assemble(("✔ ", "good"), (rel(ckpt), f"underline {C['cyan']}"), ("  ·  ", C["grey"]),
                                (session, C["cyan"]), ("  ·  ", C["grey"]), (device_label(device), "accent")))

    with console.status("[title]decoding causally (every 50 ms, past 1 s of spikes only)…", spinner="dots12",
                        spinner_style=C["magenta"]):
        decoded = [causal_decode(model, ds, rid, tr, device, std) for tr in trials]

    pos_t, pos, vel = data.cursor.timestamps, data.cursor.pos, data.cursor.vel
    # workspace geometry from the replayed trials themselves (a session-wide percentile
    # would be inflated by cursor-tracking glitches far outside the workspace)
    starts = np.array([pos[np.searchsorted(pos_t, t_d[0])] for t_d, _ in decoded])
    center = np.median(starts, axis=0)
    ends = np.array([pos[np.searchsorted(pos_t, tr["reach"][1]) - 1] for tr in trials])
    r_target = float(np.median(np.linalg.norm(ends - center, axis=1)))
    extent = r_target * 1.35
    # fit the terminal: the right-hand panels need ~46 columns, and the canvas
    # (2 columns per row keeps it square) must fit vertically with its legend
    width = int(min(64, max(36, (console.width - 46))))
    height = width // 2
    max_rows = max(14, console.height - 8)
    global STRIP_UNITS
    STRIP_UNITS = int(max(8, min(24, console.height - 18)))
    if height > max_rows:
        height = max_rows
        width = 2 * height
    label = "never-seen monkey T" if args.monkey_t else "pretrained POYO+"
    all_true, all_pred = [], []

    # when recording an SVG screenshot, animate on a separate console and only keep
    # the final frame in the recording
    from rich.console import Console

    live_console = Console() if console.record else console
    last = None
    with Live(console=live_console, refresh_per_second=30, transient=False) as live:
        for k, (trial, (t_dec, v_dec)) in enumerate(zip(trials, decoded)):
            i0 = np.searchsorted(pos_t, t_dec[0])
            start_xy = pos[i0].copy()
            dt = np.gradient(t_dec)[:, None]
            dec_xy = start_xy + np.cumsum(v_dec * dt, axis=0)
            true_idx = np.clip(np.searchsorted(pos_t, t_dec), 0, len(pos_t) - 1)
            true_xy, true_v = pos[true_idx], vel[true_idx]
            history = {"true": [], "pred": []}
            frames = np.arange(0, len(t_dec), max(1, int(round(STEP / np.median(np.diff(t_dec))))))
            t_wall = time.perf_counter()
            for f in frames:
                canvas = Canvas(width, height, extent, center=center)
                r, cx, cy = r_target, center[0], center[1]
                for a in range(8):
                    ang = np.pi / 4 * a
                    is_target = abs(np.angle(np.exp(1j * (ang - trial["angle"])))) < np.pi / 8
                    canvas.circle(cx + r * np.cos(ang), cy + r * np.sin(ang), 0.09 * r,
                                  TARGET_COLORS[trial["target"]] if is_target else C["grey"])
                canvas.circle(cx, cy, 0.09 * r, C["grey"])
                canvas.polyline(true_xy[: f + 1], C["green"])
                canvas.polyline(dec_xy[: f + 1], C["magenta"])
                canvas.mark(*true_xy[f], "●", C["green"])
                canvas.mark(*dec_xy[f], "●", C["magenta"])
                history["true"].extend(true_v[max(0, f - 4) : f + 1].tolist())
                history["pred"].extend(v_dec[max(0, f - 4) : f + 1].tolist())
                _, rate = activity_strip(data, t_dec[f])
                last = frame(canvas.render(), data, t_dec[f], trial, k, len(trials), v_dec[f], true_v[f],
                             rate, session, label, history)
                live.update(last)
                # real-time pacing
                target_wall = (t_dec[f] - t_dec[0]) / args.speed
                sleep = target_wall - (time.perf_counter() - t_wall)
                if sleep > 0:
                    time.sleep(sleep)
            all_true.append(true_v)
            all_pred.append(v_dec)
            time.sleep(0.6 / args.speed)

    if console.record and last is not None:
        console.print(last)
    total = r2(np.concatenate(all_true), np.concatenate(all_pred))
    if args.json:
        import json

        Path(args.json).write_text(json.dumps({"causal_r2": total, "session": session, "trials": len(trials)}))
    console.print(
        Panel(
            Text.assemble(
                ("Causal decoding R² over the replayed trials: ", C["fg"]), (f"{total:.3f}", score_style(total)),
                ("\nEvery prediction used only the previous second of spikes, like a real BCI.", C["grey"]),
            ),
            border_style=C["green"],
        )
    )


if __name__ == "__main__":
    main()
