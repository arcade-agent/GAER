"""Compare GAER and Arcade on GAER's cleaned file-level graphs and evaluator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Existing research modules are scripts rooted at src/, rather than a package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.clusterer import ClusterAndEval  # noqa: E402
from preprocessing.gaer_data import HeterogeneousData  # noqa: E402
from train.gae_training import train  # noqa: E402

DATASETS = {"Bash": "bash", "AS4": "archstudio"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_payload(data: HeterogeneousData) -> dict[str, object]:
    """Project exactly the cleaned GAER nodes and typed dependency pairs."""
    nodes = [
        {"id": str(file), "package": str(Path(str(file)).parent).replace("/", ".")}
        for file in data.df["File"]
    ]
    edges = [
        {"source": str(source), "target": str(target), "relation": str(relation)}
        for source, target, relation in data.df_dep[
            ["Source_File", "Target_File", "Dependency_Type"]
        ].itertuples(index=False, name=None)
    ]
    return {"schema_version": 1, "nodes": nodes, "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arcade-python", type=Path, required=True)
    parser.add_argument("--arcade-revision", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["Bash"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "comparison")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        parser.error("Set PYTHONHASHSEED to --seed before starting both runtimes")
    if args.epochs < 1 or args.clusters < 1:
        parser.error("epochs and clusters must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ClusterAndEval()
    rows: list[dict[str, object]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    preprocessing_times: dict[str, float] = {}
    provenance: dict[str, object] = {
        "gaer_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "arcade_revision": args.arcade_revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "epochs": args.epochs,
        "requested_clusters": args.clusters,
        "word2vec": {"workers": 1, "seed": 1},
        "scope": "GAER Depends 0.9.7 cleaned file-level graph; parser not evaluated",
        "evaluation": "GAER ClusterAndEval._eval including its multilabel relaxation",
        "algorithm_notes": "GAER GAT+AHC and Arcade WCA fixed k; ACDC selects its own k",
        "training": {"encoder": "gat", "hidden": 128, "dropout": 0.0, "lr": 0.001, "device": "cpu"},
        "inputs": input_hashes,
        "shared_preprocessing_seconds": preprocessing_times,
        "driver_sha256": digest(Path(__file__)),
        "worker_sha256": digest(ROOT / "benchmarks" / "arcade_worker.py"),
    }
    for python, name in (
        (Path(sys.executable), "gaer"),
        (args.arcade_python, "arcade"),
    ):
        freeze = subprocess.check_output(
            [
                str(python),
                "-c",
                "from importlib.metadata import distributions; "
                "print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version "
                "for d in distributions())))",
            ],
            text=True,
        )
        (args.out_dir / f"{name}-requirements.txt").write_text(freeze)
    for name in args.datasets:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(1)
        stem = DATASETS[name]
        node_csv, dep_csv = (
            args.data_dir / f"{stem}.csv",
            args.data_dir / f"{stem}_deps.csv",
        )
        input_hashes[name] = {
            "nodes_sha256": digest(node_csv),
            "deps_sha256": digest(dep_csv),
        }
        started = time.perf_counter()
        data = HeterogeneousData(
            pd.read_csv(node_csv),
            pd.read_csv(dep_csv),
            w2v_params={
                "vector_size": 100,
                "window": 5,
                "min_count": 5,
                "sg": 1,
                "epochs": 10,
                "max_vocab_size": 2000,
                "workers": 1,
            },
        )
        preprocessing_times[name] = time.perf_counter() - started
        nodes = data.df["File"].astype(str).tolist()
        if not 1 <= args.clusters <= len(nodes):
            parser.error(f"Invalid cluster count for {name}: {len(nodes)} nodes")
        graph_file = args.out_dir / f"{name}-graph.json"
        graph_file.write_text(json.dumps(graph_payload(data), indent=2) + "\n")
        worker_file = args.out_dir / f"{name}-arcade.json"
        subprocess.run(
            [
                str(args.arcade_python),
                str(ROOT / "benchmarks" / "arcade_worker.py"),
                "--graph",
                str(graph_file),
                "--output",
                str(worker_file),
                "--clusters",
                str(args.clusters),
            ],
            check=True,
            timeout=600,
        )
        worker = json.loads(worker_file.read_text())
        if worker["nodes"] != nodes:
            raise ValueError("Arcade output changed the file ordering")
        for algorithm, result in worker["results"].items():
            row: dict[str, object] = {
                "Dataset": name,
                "Pipeline": f"Arcade-{algorithm}",
                "Nodes": len(nodes),
                "Status": result["status"],
                "Algorithm_seconds": result["seconds"],
            }
            if result["status"] == "ok":
                row["Recovered_clusters"] = result["clusters"]
                row.update(
                    evaluator._eval(
                        np.asarray(result["labels"], dtype=int),
                        data.df["Label"].to_numpy(dtype=int),
                        nodes,
                        data.df_dep,
                        data,
                    )
                )
            else:
                row["Error"] = result["error"]
            rows.append(row)
        started = time.perf_counter()
        embeddings, _ = train(
            data,
            encoder="gat",
            epochs=args.epochs,
            hidden=128,
            dropout=0.0,
            lr=1e-3,
            device=torch.device("cpu"),
        )
        trained = time.perf_counter()
        result, labels = evaluator.run(data, embeddings, user_k=args.clusters)
        rows.append(
            {
                "Dataset": name,
                "Pipeline": "GAER-GAT-AHC",
                "Nodes": len(nodes),
                "Status": "ok",
                **result.iloc[0].to_dict(),
                "Train_seconds": trained - started,
                "Algorithm_seconds": time.perf_counter() - started,
            }
        )
        (args.out_dir / f"{name}-gaer-labels.json").write_text(json.dumps(labels, indent=2))
        # Persist after every dataset so a later failure does not erase completed work.
        columns = sorted({key for row in rows for key in row})
        with (args.out_dir / "comparison.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (args.out_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"{name}: wrote {len(rows)} rows to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
