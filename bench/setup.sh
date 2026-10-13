#!/usr/bin/env bash
# One-time setup for the benchmarks: a Python 3.12 venv with Clade and the competitor optimizers, plus pinned clones
# of τ²-bench (not on PyPI under its own name) and the official ACE prompts.
set -euo pipefail
cd "$(dirname "$0")/.."

TAU2_COMMIT=4ce7c03   # τ²-bench v1.0.1
if [ ! -d bench/vendor/tau2-bench ]; then
  git clone -q https://github.com/sierra-research/tau2-bench bench/vendor/tau2-bench
fi
git -C bench/vendor/tau2-bench checkout -q "$TAU2_COMMIT"

ACE_COMMIT=82709de    # ace-agent/ace: the ACE baseline reads its official prompts from here
if [ ! -d bench/vendor/ace ]; then
  git clone -q https://github.com/ace-agent/ace bench/vendor/ace
fi
git -C bench/vendor/ace checkout -q "$ACE_COMMIT"

[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -e ".[dev,bench]" -e bench/vendor/tau2-bench
echo "ready: .venv/bin/python -m bench.run --help"
