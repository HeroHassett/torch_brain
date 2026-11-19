#!/usr/bin/env zsh
# Re-creates the demo environment from scratch (only needed on a new machine;
# on this Mac the `torch_brain` conda env is already set up).
#
#   zsh demo/setup_env.sh
set -e
cd "${0:A:h}/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE '^torch_brain\s'; then
  conda create -y -n torch_brain python=3.11
fi
conda activate torch_brain

python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -r demo/requirements.txt

# Terminal eye candy used during the show (optional)
if command -v brew >/dev/null; then
  brew install fastfetch btop macmon glow bat eza lazygit || true
fi

python demo/check_setup.py || true
