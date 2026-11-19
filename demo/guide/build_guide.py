"""Rebuilds the showcase PDF: records each act as a screenshot, collects the numbers,
and compiles the LaTeX run sheet.

    python demo/guide/build_guide.py                # capture + compile (~3 min)
    python demo/guide/build_guide.py --no-capture   # only recompile the PDF

Needs: xelatex (MacTeX), rsvg-convert (brew install librsvg), and the demo checkpoints.
Output: POYO_Showcase_Guide.pdf at the repository root.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

GUIDE = Path(__file__).resolve().parent
REPO = GUIDE.parents[1]
IMG = GUIDE / "img"
RESULTS = GUIDE / "results"
BUILD = GUIDE / "build"
PY = sys.executable

BANNER = (
    "import sys; sys.path.insert(0, 'demo'); import common; "
    "common.banner('Perich, Miller et al. 2018 · DANDI:000688 · processed with brainsets', "
    "act='ACT 1 · MEET THE DATA')"
)

CAPTURES = [
    ("act1_banner", [PY, "-c", BANNER]),
    ("act1", [PY, "demo/01_explore_data.py"]),
    ("act2", [PY, "demo/02_model_tour.py"]),
    ("act4", [PY, "demo/03_decode.py", "--json", str(RESULTS / "decode.json")]),
    ("act5", [PY, "demo/04_live_bci.py", "--speed", "8", "--json", str(RESULTS / "bci.json")]),
    ("act7", [PY, "demo/03_decode.py", "--monkey-t", "--json", str(RESULTS / "decode_t.json")]),
    ("act7_bci", [PY, "demo/04_live_bci.py", "--monkey-t", "--speed", "8", "--json", str(RESULTS / "bci_t.json")]),
]

FIGURES = {
    "01_data_overview.png": "fig_01_data_overview.png",
    "03_decoding.png": "fig_03_decoding.png",
    "05_monkey_t_decoding.png": "fig_05_monkey_t_decoding.png",
}


def capture():
    IMG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # LINES=44: a full-screen-ish terminal height (the live BCI canvas fits the height)
    env = dict(os.environ, POYO_DEMO_WIDTH="118", TORCH_BRAIN_NO_ITERM="1", COLUMNS="118", LINES="44",
               POYO_DEMO_SVG_ASPECT="0.5")
    for name, cmd in CAPTURES:
        svg = IMG / f"{name}.svg"
        print(f"· capturing {name}: {' '.join(cmd[1:3])} …", flush=True)
        subprocess.run(cmd, cwd=REPO, env=dict(env, POYO_DEMO_SVG=str(svg)), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text = svg.read_text()
        # render with the iTerm2 profile's own font (Iosevka Nerd Font Mono, has braille glyphs)
        text = text.replace('font-family: "Fira Code"', 'font-family: "Iosevka Nerd Font Mono"')
        text = text.replace("font-family: Fira Code, monospace", 'font-family: "Iosevka Nerd Font Mono", monospace')
        svg.write_text(text)
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(IMG / f"{name}.pdf"), str(svg)], check=True)
    for src, dst in FIGURES.items():
        if (REPO / "demo" / "figures" / src).exists():
            shutil.copyfile(REPO / "demo" / "figures" / src, IMG / dst)


def write_numbers():
    load = lambda n: json.loads((RESULTS / n).read_text())
    dec, dec_t, bci = load("decode.json"), load("decode_t.json"), load("bci.json")
    extra = json.loads((GUIDE / "timings.json").read_text())
    best_session, best = max(dec["r2"].items(), key=lambda kv: kv[1])
    best_session = best_session.replace("_", r"\_")
    numbers = {
        "PretrainAvg": f"{dec['average_r2']:.2f}",
        "PretrainBest": f"{best:.2f} ({best_session})",
        "MonkeyTAvg": f"{dec_t['average_r2']:.2f}",
        "BCIRtwo": f"{bci['causal_r2']:.2f}",
        "DecodeSeconds": f"{max(5, round(dec['seconds_elapsed'] / 5) * 5)}",
        "DecodeWindows": f"{dec['windows']:,}".replace(",", "{,}"),
        "DecodeMinutes": f"{dec['minutes_decoded']:.0f}",
        **extra,
    }
    (GUIDE / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in numbers.items())
    )

    names = {"c": "Monkey C", "m": "Monkey M", "j": "Monkey J", "t": "Monkey T"}
    rows = []
    for session, r2 in sorted(dec["r2"].items(), key=lambda kv: -kv[1]):
        task = "center-out" if "center_out" in session else "random-target"
        date = re.search(r"_(\d{8})_", session).group(1)
        bar = f"\\textcolor{{lmauve}}{{\\rule{{{max(r2, 0) * 4:.2f}cm}}{{1.6ex}}}}"
        rows.append(f"{names[session[0]]} & {date[:4]}-{date[4:6]}-{date[6:]} & {task} & "
                    f"{dec['units'][session]} & {r2:.3f} & {bar} \\\\")
    table = (
        "\\begin{center}\\renewcommand{\\arraystretch}{1.12}\n"
        "\\begin{tabular}{l l l r r l}\\toprule\n"
        "\\textbf{Subject} & \\textbf{Date} & \\textbf{Task} & \\textbf{Neurons} & \\textbf{R²} & \\\\ \\midrule\n"
        + "\n".join(rows)
        + f"\n\\midrule \\textbf{{Average}} & & & & \\textbf{{{dec['average_r2']:.3f}}} & \\\\ \\bottomrule\n"
        "\\end{tabular}\\end{center}\n"
    )
    (GUIDE / "r2_table.tex").write_text(table)


def compile_pdf():
    BUILD.mkdir(exist_ok=True)
    for _ in range(2):  # twice for the table of contents
        subprocess.run(["xelatex", "-interaction=nonstopmode", f"-output-directory={BUILD}", "showcase_guide.tex"],
                       cwd=GUIDE, stdout=subprocess.DEVNULL, check=False)
    pdf = BUILD / "showcase_guide.pdf"
    log = (BUILD / "showcase_guide.log").read_text(errors="ignore")
    errors = [line for line in log.splitlines() if line.startswith("! ")]
    if errors:
        print("LaTeX errors:\n" + "\n".join(errors[:10]))
    shutil.copyfile(pdf, REPO / "POYO_Showcase_Guide.pdf")
    print(f"✔ wrote {REPO / 'POYO_Showcase_Guide.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args()
    if not args.no_capture:
        capture()
    write_numbers()
    compile_pdf()
