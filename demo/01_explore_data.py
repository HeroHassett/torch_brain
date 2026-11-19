"""Act 1 · Meet the data: motor-cortex spikes recorded while monkeys reach.

    python demo/01_explore_data.py            # full tour
    python demo/01_explore_data.py --fast     # skip the full 111-session scan
"""

import argparse
from collections import defaultdict

import common  # noqa: F401  (sets up paths / env)
from common import (
    BRAINSET,
    C,
    DATA_ROOT,
    FIG_DIR,
    MONKEYS,
    banner,
    console,
    human,
    section,
    show_figure,
)

import h5py
import numpy as np
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from temporaldata import Data

FEATURED = "c_20150629_center_out_reaching"


def scan_brainset():
    files = sorted((DATA_ROOT / BRAINSET).glob("*.h5"))
    rows = []
    with Progress(
        SpinnerColumn("dots12", style=C["magenta"]),
        TextColumn("[title]scanning recordings"),
        BarColumn(complete_style=C["magenta"], finished_style=C["green"]),
        MofNCompleteColumn(),
        TextColumn("[muted]{task.fields[name]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("scan", total=len(files), name="")
        for f in files:
            progress.update(task, name=f.stem)
            with h5py.File(f, "r") as h:
                d = Data.from_hdf5(h, lazy=True)
                dur = float(d.domain.end[-1] - d.domain.start[0])
                rows.append(
                    dict(
                        session=f.stem,
                        monkey=f.stem[0],
                        task="center-out" if "center_out" in f.stem else "random-target",
                        units=len(d.units.id),
                        spikes=len(d.spikes.timestamps),
                        duration=dur,
                        mb=f.stat().st_size / 2**20,
                    )
                )
            progress.advance(task)
    return rows


def brainset_tables(rows):
    by_monkey = defaultdict(list)
    for r in rows:
        by_monkey[r["monkey"]].append(r)

    table = Table(
        title="perich_miller_population_2018 · 4 monkeys · motor cortex (M1 / PMd)",
        title_style="title",
        box=box.ROUNDED,
        border_style=C["grey"],
        header_style="accent",
        show_footer=True,
        footer_style="bold",
    )
    total = dict(n=0, spikes=0, hours=0.0, mb=0.0)
    table.add_column("subject", footer="total")
    table.add_column("sessions", justify="right")
    table.add_column("center-out / random", justify="center")
    table.add_column("neurons / session", justify="center")
    table.add_column("spikes", justify="right")
    table.add_column("hours", justify="right")
    table.add_column("on disk", justify="right")
    for key in ["c", "m", "j", "t"]:
        rs = by_monkey[key]
        name, color = MONKEYS[key]
        co = sum(r["task"] == "center-out" for r in rs)
        units = [r["units"] for r in rs]
        spikes = sum(r["spikes"] for r in rs)
        hours = sum(r["duration"] for r in rs) / 3600
        mb = sum(r["mb"] for r in rs)
        total["n"] += len(rs)
        total["spikes"] += spikes
        total["hours"] += hours
        total["mb"] += mb
        held_out = " [muted](held out)[/]" if key == "t" else ""
        table.add_row(
            Text.from_markup(f"[bold {color}]●[/] {name}{held_out}"),
            str(len(rs)),
            f"{co} / {len(rs) - co}",
            f"{min(units)} – {max(units)}",
            human(spikes),
            f"{hours:.1f}",
            f"{mb / 1024:.2f} GB" if mb > 1024 else f"{mb:.0f} MB",
        )
    table.columns[1].footer = str(total["n"])
    table.columns[4].footer = human(total["spikes"])
    table.columns[5].footer = f"{total['hours']:.1f}"
    table.columns[6].footer = f"{total['mb'] / 1024:.2f} GB"
    console.print(table)


def load(session):
    f = h5py.File(DATA_ROOT / BRAINSET / f"{session}.h5", "r")
    return Data.from_hdf5(f, lazy=False)


def terminal_raster(d, t0, seconds=10.0, n_units=22, bins_per_s=None):
    """A truecolour spike-count heatmap drawn with unicode blocks, in the terminal."""
    if bins_per_s is None:  # fit the terminal: 10 bins/s on wide windows
        bins_per_s = int(max(4, min(10, (console.width - 12) // seconds)))
    ts, ui = d.spikes.timestamps, d.spikes.unit_index
    sel = (ts >= t0) & (ts < t0 + seconds)
    ts, ui = ts[sel], ui[sel]
    counts = np.bincount(ui, minlength=len(d.units.id))
    top = np.argsort(-counts)[:n_units]
    nb = int(seconds * bins_per_s)
    grid = np.zeros((n_units, nb))
    for row, u in enumerate(top):
        h, _ = np.histogram(ts[ui == u], bins=nb, range=(t0, t0 + seconds))
        grid[row] = h
    grid = grid / max(grid.max(), 1)

    def mix(a, b, x):
        a = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
        b = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{int(a[i] + (b[i] - a[i]) * x):02x}" for i in range(3))

    lines = Text()
    for row, u in enumerate(top):
        lines.append(f" unit {u:>3} ", style=C["grey"])
        for x in grid[row]:
            color = mix("#24273a", C["magenta"], x) if x < 0.5 else mix(
                C["magenta"], C["yellow"], (x - 0.5) * 2
            )
            lines.append("█", style=color)
        lines.append("\n")
    axis = Text(" " * 10)
    for s in range(int(seconds)):
        axis.append(f"{s}s".ljust(bins_per_s), style=C["grey"])
    lines.append(axis)
    return lines


def make_figure(d, t0, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg, grid = "#24273a", "#cad3f5", "#363a4f"
    plt.rcParams.update(
        {
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "axes.edgecolor": grid,
            "axes.labelcolor": fg,
            "text.color": fg,
            "xtick.color": fg,
            "ytick.color": fg,
            "font.size": 11,
        }
    )
    fig = plt.figure(figsize=(15, 6.5), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[2.2, 1], wspace=0.15, hspace=0.12)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_f = fig.add_subplot(gs[:, 1])

    seconds = 10.0
    ts, ui = d.spikes.timestamps, d.spikes.unit_index
    sel = (ts >= t0) & (ts < t0 + seconds)
    ax_r.scatter(ts[sel] - t0, ui[sel], s=2.5, c=C["cyan"], marker="|", linewidths=0.8)
    ax_r.set_ylabel("neuron")
    ax_r.set_title(f"{FEATURED}  ·  every tick is one spike", loc="left", color=C["blue"], fontsize=12)
    plt.setp(ax_r.get_xticklabels(), visible=False)

    ct, vel = d.cursor.timestamps, d.cursor.vel
    sel = (ct >= t0) & (ct < t0 + seconds)
    ax_v.plot(ct[sel] - t0, vel[sel, 0], color=C["magenta"], lw=1.6, label="v$_x$")
    ax_v.plot(ct[sel] - t0, vel[sel, 1], color=C["yellow"], lw=1.6, label="v$_y$")
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("cursor velocity")
    ax_v.legend(loc="upper right", frameon=False, ncol=2)

    # center-out "flower": every reach of the session, coloured by target
    palette = [C["red"], C["orange"], C["yellow"], C["green"], C["cyan"], C["blue"], C["magenta"], "#f5bde6"]
    rp = d.movement_phases.reach_period
    tr = d.trials
    pos = d.cursor.pos
    for s, e in zip(rp.start, rp.end):
        k = np.searchsorted(tr.start, s, side="right") - 1
        tid = tr.target_id[k]
        if np.isnan(tid):
            continue
        m = (ct >= s) & (ct <= e)
        ax_f.plot(pos[m, 0], pos[m, 1], color=palette[int(tid)], lw=1.0, alpha=0.75)
    ax_f.set_aspect("equal")
    ax_f.set_title("hand trajectories · 8 reach targets", color=C["blue"], fontsize=12)
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    for ax in (ax_r, ax_v, ax_f):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(path, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def tokens_panel(d, t0):
    ts, ui = d.spikes.timestamps, d.spikes.unit_index
    sel = (ts >= t0) & (ts < t0 + 1.0)
    n_spikes = int(sel.sum())
    n_units = len(d.units.id)
    table = Table(box=box.SIMPLE_HEAD, header_style="accent", border_style=C["grey"], padding=(0, 1))
    table.add_column("#", style=C["grey"], justify="right")
    table.add_column("unit (neuron id)", style=C["cyan"])
    table.add_column("time (s)", style=C["yellow"], justify="right")
    for i, (u, t) in enumerate(zip(ui[sel][:9], ts[sel][:9])):
        uid = d.units.id[u]
        uid = uid.decode() if isinstance(uid, bytes) else str(uid)
        table.add_row(str(i), uid.replace("group_electrode_group_", ""), f"{t - t0:.4f}")
    table.add_row("…", "…", "…")

    story = Text.assemble(
        ("POYO's key idea: ", "accent"),
        ("every spike is a token.\n\n", C["fg"]),
        ("No binning, no fixed channel order: a token is just\n", C["grey"]),
        ("  (which neuron fired, when it fired)\n\n", C["cyan"]),
        ("For one 1-second context window:\n", C["fg"]),
        (f"  • {n_spikes:>5} spike tokens\n", C["yellow"]),
        (f"  • {2 * n_units:>5} start/end tokens (one pair per neuron)\n", C["yellow"]),
        ("  •   128 latent tokens  (16 every 125 ms)\n", C["magenta"]),
        ("  •   100 output queries (cursor velocity @ 100 Hz)\n\n", C["green"]),
        ("Neurons get learned embeddings, so new recordings\n", C["grey"]),
        ("with different neurons plug straight in.", C["grey"]),
    )
    return Columns(
        [
            Panel(table, title="first spike tokens of the window", border_style=C["grey"], title_align="left"),
            Panel(story, title="tokenization", border_style=C["grey"], title_align="left"),
        ],
        expand=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip the full brainset scan")
    args = parser.parse_args()

    banner("Perich, Miller et al. 2018 · DANDI:000688 · processed with brainsets", act="ACT 1 · MEET THE DATA")

    if not args.fast:
        section("The brainset on disk", "🗂 ")
        rows = scan_brainset()
        brainset_tables(rows)

    with console.status("[title]loading a recording…", spinner="bouncingBall", spinner_style=C["magenta"]):
        d = load(FEATURED)
    desc = str(d.brainset.description)
    console.print(
        Panel(
            Text(desc, style=C["fg"]),
            title=f"[title]{d.brainset.id}[/]",
            subtitle=f"[link={d.brainset.source}][cyan]{d.brainset.source}[/][/link]",
            border_style=C["grey"],
            padding=(1, 2),
        )
    )

    # a moment with clear reaching activity inside the recording
    t0 = float(d.trials.start[40])

    section(f"Live from motor cortex · {FEATURED}", "⚡")
    console.print(terminal_raster(d, t0))
    console.print(
        Text("   ↑ spike counts of the 22 most active neurons over 10 s, drawn in truecolour", style="muted")
    )

    section("Spikes → tokens", "🧩")
    console.print(tokens_panel(d, t0))

    section("The same 10 seconds, in high resolution", "🖼 ")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "01_data_overview.png"
    with console.status("[title]rendering figure…", spinner="dots12", spinner_style=C["magenta"]):
        make_figure(d, t0, path)
    show_figure(path, "spike raster · cursor velocity · all reaches of the session")

    console.print(
        Panel(
            Text.assemble(
                ("Goal: ", "accent"),
                ("read the spikes, predict where the hand is moving (cursor velocity), ", C["fg"]),
                ("in real time, for any monkey and any set of neurons.", f"bold {C['green']}"),
            ),
            border_style=C["green"],
        )
    )


if __name__ == "__main__":
    main()
