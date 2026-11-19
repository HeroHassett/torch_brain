#!/usr/bin/env zsh
# POYO+ terminal showcase · autopilot
# Runs every act back to back and waits for <Enter> between them.
#
#   conda activate torch_brain
#   zsh demo/showcase.sh            # all acts
#   zsh demo/showcase.sh 3          # start at act 3
#   SKIP_TRAIN=1 zsh demo/showcase.sh   # skip the two training acts

cd "${0:A:h}/.." || exit 1

if [[ "$CONDA_DEFAULT_ENV" != "torch_brain" ]]; then
  print -P "%F{red}✘ activate the environment first:%f  conda activate torch_brain"
  exit 1
fi

START=${1:-0}

act() {
  local num=$1; shift
  local title=$1; shift
  (( num < START )) && return 0
  print
  print -P "%F{8}────────────────────────────────────────────────────────────────%f"
  print -P "  %B%F{magenta}ACT ${num}%f%b  %F{blue}${title}%f"
  print -P "  %F{8}\$ ${*}%f"
  print -P "%F{8}────────────────────────────────────────────────────────────────%f"
  read -s "?  ⏎  press Enter to run (Ctrl-C to quit) "
  print
  eval "$@"
}

act 0 "Hello, machine"                  fastfetch
act 1 "Meet the data"                   python demo/01_explore_data.py
act 2 "Inside POYO+"                    python demo/02_model_tour.py
[[ -z "$SKIP_TRAIN" ]] && \
act 3 "Train POYO+ live, from scratch"  python train.py --config-name demo_live
act 4 "Decode held-out trials"          python demo/03_decode.py
act 5 "Live brain-computer interface"   python demo/04_live_bci.py
[[ -z "$SKIP_TRAIN" ]] && \
act 6 "Transfer to a never-seen monkey" python finetune.py --config-name demo_finetune
act 7 "Decode monkey T"                 python demo/03_decode.py --monkey-t
act 7 "Live BCI on monkey T"            python demo/04_live_bci.py --monkey-t
act 8 "Encore: the test suite"          python -m pytest tests/test_poyo_plus.py tests/test_apple_silicon.py -v -W ignore -p no:cacheprovider

print -P "\n%B%F{green}🎉 That's the show!%f%b"
printf '\e]6;1;bg;*;default\a\e]1337;SetBadgeFormat=\a'   # reset iTerm2 tab colour + badge
