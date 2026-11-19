"""Small helpers for rich terminal output.

Everything in here degrades gracefully: the iTerm2-specific escape sequences are only
emitted when running inside iTerm2 (also through tmux), and are silently skipped
everywhere else (plain terminals, CI logs, files, ...).

iTerm2 proprietary escape codes reference: https://iterm2.com/documentation-escape-codes.html
"""

import base64
import os
import sys
from pathlib import Path
from typing import Optional, Union

import lightning as L

# Catppuccin Macchiato palette, used across the CLI outputs so that everything
# matches the iTerm2 colour scheme (https://catppuccin.com/palette).
PALETTE = {
    "fg": "#cad3f5",
    "red": "#ed8796",
    "green": "#a6da95",
    "yellow": "#eed49f",
    "blue": "#8aadf4",
    "magenta": "#c6a0f6",
    "cyan": "#8bd5ca",
    "orange": "#f5a97f",
    "grey": "#6e738d",
}


def in_iterm2() -> bool:
    """Returns True when the process is attached to an iTerm2 terminal."""
    if os.environ.get("TORCH_BRAIN_NO_ITERM", "0") == "1":
        return False
    return (
        os.environ.get("TERM_PROGRAM") == "iTerm.app"
        or os.environ.get("LC_TERMINAL") == "iTerm2"
    )


def _write_osc(payload: str) -> None:
    """Writes an OSC escape sequence to the terminal (wrapping it for tmux)."""
    if not in_iterm2():
        return
    seq = f"\033]{payload}\a"
    if os.environ.get("TMUX"):
        # tmux passthrough: escape every ESC and wrap in a DCS sequence
        seq = "\033Ptmux;" + seq.replace("\033", "\033\033") + "\033\\"
    stream = sys.__stdout__
    try:
        stream.write(seq)
        stream.flush()
    except Exception:  # never crash a training run because of eye candy
        pass


def show_image(
    path: Union[str, Path], width: str = "auto", height: str = "auto"
) -> bool:
    """Displays an image inline in the terminal (iTerm2 inline images protocol).

    Returns True if the image was displayed, False otherwise (the caller can then
    print the path instead).
    """
    path = Path(path)
    if not in_iterm2() or not path.exists():
        return False
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    name = base64.b64encode(path.name.encode()).decode("ascii")
    _write_osc(
        f"1337;File=name={name};size={path.stat().st_size};inline=1;"
        f"width={width};height={height};preserveAspectRatio=1:{data}"
    )
    sys.__stdout__.write("\n")
    sys.__stdout__.flush()
    return True


def notify(message: str) -> None:
    """Posts a macOS notification through iTerm2 (OSC 9)."""
    _write_osc(f"9;{message}")


def fireworks() -> None:
    """🎆 Asks iTerm2 to play its fireworks animation at the cursor position."""
    _write_osc("1337;RequestAttention=fireworks")


def set_badge(text: Optional[str]) -> None:
    """Sets (or clears, with None/"") the translucent badge shown in the iTerm2 pane."""
    encoded = base64.b64encode((text or "").encode()).decode("ascii")
    _write_osc(f"1337;SetBadgeFormat={encoded}")


def set_tab_color(hex_color: Optional[str]) -> None:
    """Sets the iTerm2 tab colour (e.g. "#a6da95"), or resets it with None."""
    if hex_color is None:
        _write_osc("6;1;bg;*;default")
        return
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    for channel, value in zip(("red", "green", "blue"), (r, g, b)):
        _write_osc(f"6;1;bg;{channel};brightness;{value}")


def set_mark() -> None:
    """Drops an iTerm2 mark (jump back to it with Cmd+Shift+Up)."""
    _write_osc("1337;SetMark")


class TerminalNotifier(L.Callback):
    r"""Lightning callback that makes long runs visible from anywhere on the desktop:
    colours the iTerm2 tab while training, then posts a notification and plays
    iTerm2's fireworks when the run finishes. No-op outside iTerm2.
    """

    def __init__(self, run_name: str = "POYO+", badge: Optional[str] = None):
        self.run_name = run_name
        self.badge = badge

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            set_tab_color(PALETTE["yellow"])
            if self.badge:
                set_badge(self.badge)

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metric = trainer.callback_metrics.get("average_val_metric")
        suffix = f" (val R² = {float(metric):.3f})" if metric is not None else ""
        set_tab_color(PALETTE["green"])
        notify(f"🧠 {self.run_name} finished training{suffix}")
        fireworks()

    def on_exception(self, trainer, pl_module, exception):
        set_tab_color(PALETTE["red"])
