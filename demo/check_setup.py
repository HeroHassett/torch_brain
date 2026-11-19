"""Preflight check for the POYO+ terminal demo: run this before going on stage.

    python demo/check_setup.py
"""

import importlib
import importlib.metadata
import os
import shutil
import sys

import common  # noqa: F401
from common import BRAINSET, C, CKPT_DIR, DATA_ROOT, FINETUNED_CKPT, PRETRAINED_CKPT, REPO, console, rel

from rich import box
from rich.table import Table
from rich.text import Text

from torch_brain.utils import terminal as term

rows = []


def check(name, ok, detail="", optional=False):
    rows.append((name, ok, detail, optional))


# python environment
env = os.environ.get("CONDA_DEFAULT_ENV", "")
check("conda env 'torch_brain' active", env == "torch_brain", f"CONDA_DEFAULT_ENV={env or '∅'} · {sys.executable}")
for mod in ["torch", "lightning", "hydra", "temporaldata", "torchmetrics", "rich", "plotext", "matplotlib", "tensorboard"]:
    try:
        importlib.import_module(mod)
        dist = {"hydra": "hydra-core"}.get(mod, mod)
        check(f"python package: {mod}", True, importlib.metadata.version(dist))
    except Exception as e:  # pragma: no cover
        check(f"python package: {mod}", False, str(e))

import torch  # noqa: E402

check("Apple-Silicon GPU (MPS)", torch.backends.mps.is_available(), f"torch {torch.__version__}", optional=True)

# data + checkpoints
n_sessions = len(list((DATA_ROOT / BRAINSET).glob("*.h5")))
check("processed brainset", n_sessions == 111, f"{n_sessions} / 111 sessions in {rel(DATA_ROOT / BRAINSET)}")
check("pretrained checkpoint", PRETRAINED_CKPT.exists(), rel(PRETRAINED_CKPT))
check(
    "monkey-T checkpoint",
    FINETUNED_CKPT.exists(),
    rel(FINETUNED_CKPT) + ("" if FINETUNED_CKPT.exists() else "  (created by: python finetune.py --config-name demo_finetune)"),
    optional=True,
)

# terminal goodies
check("running inside iTerm2", term.in_iterm2(), f"TERM_PROGRAM={os.environ.get('TERM_PROGRAM', '∅')} (inline images, fireworks, badges)", optional=True)
check("truecolor terminal", os.environ.get("COLORTERM") in ("truecolor", "24bit"), f"COLORTERM={os.environ.get('COLORTERM', '∅')}", optional=True)
for tool in ["fastfetch", "glow", "bat", "eza", "zoxide", "starship", "btop", "macmon", "lazygit", "nvim", "fzf"]:
    path = shutil.which(tool)
    check(f"cli tool: {tool}", path is not None, path or "not installed", optional=True)

table = Table(title="🧠 POYO+ demo · preflight", title_style="title", box=box.ROUNDED, border_style=C["grey"], header_style="accent")
table.add_column("")
table.add_column("check")
table.add_column("details", style=C["grey"])
required_ok = True
for name, ok, detail, optional in rows:
    mark = Text("✔", style="good") if ok else Text("!" if optional else "✘", style="warn" if optional else "bad")
    if not ok and not optional:
        required_ok = False
    table.add_row(mark, name, detail)
console.print(table)
if required_ok:
    console.print(Text("All required checks passed. Break a leg! 🎭", style="good"))
else:
    console.print(Text("Some required checks failed — see the ✘ rows above.", style="bad"))
    sys.exit(1)
