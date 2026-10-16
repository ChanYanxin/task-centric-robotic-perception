#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[test,plots]'
tc-perception generate
tc-perception benchmark
tc-perception render --mode event
pytest -q
printf '\nReady: docs/assets/event_demo.gif and results/benchmark_summary.json\n'
