# Shared-graph Arcade comparison

This optional experiment compares GAER-GAT-AHC and the Python `arcade-agent`
WCA/ACDC implementations on the **same cleaned Depends 0.9.7 file-level graph**.
It evaluates all valid partitions with GAER's existing evaluator, including its
multilabel relaxation and movement-cost A2A metric. Arcade's similarly named
Jaccard architecture similarity is not substituted for that metric.

GAER requires Python 3.11/NumPy 1.x; Arcade requires Python 3.12+/NumPy 2.x.
Use separate virtual environments. The worker receives file paths, directory
packages and typed dependency pairs, without ground-truth module labels.
GAER additionally uses its usual text/directory features. Dependency counts are
retained for TurboMQ evaluation; neither encoder nor Arcade exchange edges use
those counts as training weights. This comparison isolates recovery algorithms
and does **not** evaluate either tool's source parser or reproduce the complete
paper experiment. Fixed `k=10` is independent of ground truth; ACDC determines
its own cluster count. Invalid/overlapping partitions are recorded without scores.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r benchmarks/requirements-cpu.txt
python3.12 -m venv .arcade-venv
.arcade-venv/bin/pip install 'arcade-agent @ git+https://github.com/arcade-agent/arcade-agent.git@503e998a2193b01626a587e5c64a0631627a7dc0'
git lfs pull --include='data/bash.csv,data/bash_deps.csv,data/archstudio.csv,data/archstudio_deps.csv'
PYTHONHASHSEED=0 .venv/bin/python benchmarks/compare.py \
  --arcade-python "$PWD/.arcade-venv/bin/python" \
  --arcade-revision 503e998a2193b01626a587e5c64a0631627a7dc0 \
  --datasets Bash AS4 --seed 0 --epochs 30 --clusters 10
```

Results contain input SHA256s, source revisions, both dependency snapshots,
parameters, predictions, invalid-partition details, and CSV metrics. The shared
preprocessing time is separate from algorithm time. GAER algorithm time includes
training and clustering; Arcade time covers clustering. Use multiple seeds and
datasets before drawing comparative effectiveness/performance conclusions.

Worker contract checks run with `.arcade-venv/bin/python -m pytest benchmarks/`.
The GitHub workflow runs the same CPU smoke comparison and uploads raw artifacts.
