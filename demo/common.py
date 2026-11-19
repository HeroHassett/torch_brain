"""Shared helpers for the POYO+ terminal demo (console theme, banner, model I/O)."""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import logging
import warnings

warnings.filterwarnings("ignore")
# keep the demo output clean (e.g. torch_brain's "data leakage check disabled" notes)
logging.getLogger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from torch_brain.utils import terminal as term  # noqa: E402
from torch_brain.utils.terminal import PALETTE as C  # noqa: E402

DATA_ROOT = REPO / "data" / "processed"
BRAINSET = "perich_miller_population_2018"
FIG_DIR = REPO / "demo" / "figures"
CKPT_DIR = REPO / "demo" / "checkpoints"
PRETRAINED_CKPT = CKPT_DIR / "poyo_plus_demo.ckpt"
FINETUNED_CKPT = CKPT_DIR / "poyo_plus_monkey_t.ckpt"

MONKEYS = {
    "c": ("Monkey C", C["blue"]),
    "m": ("Monkey M", C["magenta"]),
    "j": ("Monkey J", C["orange"]),
    "t": ("Monkey T", C["green"]),
}

_SVG_PATH = os.environ.get("POYO_DEMO_SVG")  # set to record the act as an SVG screenshot

console = Console(
    theme=Theme(
        {
            "title": f"bold {C['blue']}",
            "accent": f"bold {C['magenta']}",
            "good": f"bold {C['green']}",
            "warn": f"bold {C['yellow']}",
            "bad": f"bold {C['red']}",
            "muted": C["grey"],
            "cyan": C["cyan"],
            "orange": C["orange"],
        }
    ),
    highlight=False,
    record=bool(_SVG_PATH),
    width=int(os.environ["POYO_DEMO_WIDTH"]) if "POYO_DEMO_WIDTH" in os.environ else None,
)

if _SVG_PATH:
    import atexit

    from rich.terminal_theme import TerminalTheme

    _CATPPUCCIN_MACCHIATO = TerminalTheme(
        (36, 39, 58),  # background
        (202, 211, 244),  # foreground
        [(73, 77, 100), (237, 135, 150), (166, 218, 149), (238, 212, 159),
         (138, 173, 244), (245, 189, 230), (139, 213, 202), (184, 192, 224)],
        [(91, 96, 120), (237, 135, 150), (166, 218, 149), (238, 212, 159),
         (138, 173, 244), (245, 189, 230), (139, 213, 202), (165, 173, 203)],
    )
    atexit.register(
        lambda: console.save_svg(
            _SVG_PATH,
            title="iTerm2 · POYO+ demo",
            theme=_CATPPUCCIN_MACCHIATO,
            font_aspect_ratio=float(os.environ.get("POYO_DEMO_SVG_ASPECT", "0.61")),
        )
    )

_LOGO = [
    "██████╗  ██████╗ ██╗   ██╗ ██████╗          ",
    "██╔══██╗██╔═══██╗╚██╗ ██╔╝██╔═══██╗   ██╗   ",
    "██████╔╝██║   ██║ ╚████╔╝ ██║   ██║ ██████╗ ",
    "██╔═══╝ ██║   ██║  ╚██╔╝  ██║   ██║ ╚═██╔═╝ ",
    "██║     ╚██████╔╝   ██║   ╚██████╔╝   ╚═╝   ",
    "╚═╝      ╚═════╝    ╚═╝    ╚═════╝          ",
]
_GRADIENT = [C["blue"], C["blue"], C["cyan"], C["magenta"], C["magenta"], C["red"]]


def banner(subtitle: str, act: str = None):
    """Prints the POYO+ logo with a Catppuccin gradient and a subtitle."""
    logo = Text()
    for line, color in zip(_LOGO, _GRADIENT):
        logo.append(line + "\n", style=f"bold {color}")
    caption = Text.assemble(
        ("🧠 torch_brain", f"bold {C['fg']}"),
        ("  ·  ", C["grey"]),
        ("transformers that read neural spikes", f"italic {C['cyan']}"),
    )
    parts = [Align.center(logo), Align.center(caption)]
    if act:
        parts.append(Text())
        parts.append(Align.center(Text(act, style=f"bold {C['yellow']}")))
    parts.append(Align.center(Text(subtitle, style=C["grey"])))
    console.print(
        Panel(Group(*parts), box=box.HEAVY, border_style=C["grey"], padding=(1, 2))
    )
    term.set_badge(f"🧠 POYO+\n{act or ''}".strip())
    term.set_mark()


def section(title: str, emoji: str = "▶"):
    console.print()
    console.print(Rule(Text(f" {emoji}  {title} ", style="title"), style=C["grey"]))
    console.print()


def show_figure(path: Path, caption: str = None, width: str = "100%"):
    """Shows a figure inline (iTerm2), or tells the user where it is."""
    if caption:
        console.print(Text(f"🖼  {caption}", style="accent"))
    if not term.show_image(path, width=width):
        console.print(
            Text.assemble(
                ("   saved to ", C["grey"]),
                (str(path.relative_to(REPO)), f"underline {C['cyan']}"),
                ("  (open it with: ", C["grey"]),
                (f"open {path.relative_to(REPO)}", C["yellow"]),
                (")", C["grey"]),
            )
        )


def pick_device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device) -> str:
    if device.type == "mps":
        try:
            import subprocess

            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            chip = "Apple Silicon"
        return f"{chip} GPU (Metal / MPS)"
    if device.type == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    return "CPU"


def load_model(ckpt_path: Path, device):
    """Rebuilds the POYO+ model from a Lightning checkpoint (config + weights)."""
    import hydra
    import torch
    from omegaconf import OmegaConf

    from torch_brain.registry import MODALITIY_REGISTRY

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = OmegaConf.create(ckpt["hyper_parameters"])
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITIY_REGISTRY)
    state_dict = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, cfg, ckpt


def rel(path) -> str:
    """Path relative to the repo when possible (for display)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def score_style(value: float) -> str:
    if value >= 0.8:
        return "good"
    if value >= 0.5:
        return "warn"
    return "bad"


def score_bar(value: float, width: int = 24) -> Text:
    n = int(round(max(0.0, min(1.0, value)) * width))
    return Text("█" * n + "░" * (width - n), style=score_style(value))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
