"""Run Arcade algorithms in a separate Python 3.12+ environment.

The exchange graph contains only files, directories and dependency relations.
Ground-truth modules stay in the evaluator process.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

from arcade_agent.algorithms.acdc import acdc
from arcade_agent.algorithms.architecture import Architecture
from arcade_agent.algorithms.clustering import wca
from arcade_agent.parsers.graph import DependencyGraph, Edge, Entity


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected an object with string keys")
    return dict(value)


def text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty string")
    return value


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected an array")
    return list(value)


def load_graph(raw: object) -> DependencyGraph:
    value = mapping(raw)
    if set(value) != {"schema_version", "nodes", "edges"} or value["schema_version"] != 1:
        raise ValueError("Unsupported graph schema; ground truth must not be exported")
    graph = DependencyGraph()
    for item in sequence(value["nodes"]):
        node = mapping(item)
        if set(node) != {"id", "package"}:
            raise ValueError("Nodes must contain only id and directory-derived package")
        key = text(node["id"])
        package = node["package"]
        if not isinstance(package, str):
            raise ValueError("Package must be a string")
        if key in graph.entities:
            raise ValueError(f"Duplicate node: {key}")
        graph.entities[key] = Entity(
            fqn=key,
            name=Path(key).stem,
            package=package,
            file_path=key,
            kind="module",
            language="dataset",
        )
        graph.packages.setdefault(package, []).append(key)
    if not graph.entities:
        raise ValueError("Cannot cluster an empty graph")
    for item in sequence(value["edges"]):
        edge = mapping(item)
        source, target = text(edge["source"]), text(edge["target"])
        if source not in graph.entities or target not in graph.entities:
            raise ValueError(f"Unknown edge endpoint: {source} -> {target}")
        graph.edges.append(Edge(source, target, text(edge["relation"])))
    return graph


def partition_labels(architecture: Architecture, nodes: list[str]) -> list[int]:
    """Reject overlapping/incomplete partitions instead of silently overwriting."""
    expected = set(nodes)
    membership: dict[str, int] = {}
    for label, component in enumerate(architecture.components):
        if not component.entities:
            raise ValueError("Empty recovered component")
        for node in component.entities:
            if node not in expected:
                raise ValueError(f"Unknown recovered entity: {node}")
            if node in membership:
                raise ValueError(f"Entity belongs to multiple components: {node}")
            membership[node] = label
    if set(membership) != expected:
        raise ValueError(f"Missing {len(expected - set(membership))} recovered entities")
    return [membership[node] for node in nodes]


def run(graph: DependencyGraph, clusters: int) -> dict[str, object]:
    if not 1 <= clusters <= graph.num_entities:
        raise ValueError("Cluster count must be between 1 and the node count")
    nodes = list(graph.entities)
    results: dict[str, object] = {}
    for algorithm in ("wca", "acdc"):
        started = time.perf_counter()
        architecture = wca(graph, num_clusters=clusters) if algorithm == "wca" else acdc(graph)
        elapsed = time.perf_counter() - started
        try:
            labels = partition_labels(architecture, nodes)
        except ValueError as error:
            results[algorithm] = {
                "status": "invalid_partition",
                "error": str(error),
                "seconds": elapsed,
            }
        else:
            results[algorithm] = {
                "status": "ok",
                "labels": labels,
                "clusters": len(architecture.components),
                "seconds": elapsed,
            }
    return {
        "nodes": nodes,
        "results": results,
        "python": platform.python_version(),
        "arcade_agent_version": version("arcade-agent"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=10)
    args = parser.parse_args()
    result = run(load_graph(json.loads(args.graph.read_text())), args.clusters)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
