import pytest
from arcade_agent.algorithms.architecture import Architecture, Component
from arcade_worker import load_graph, partition_labels, run


def test_rejects_ground_truth_in_exchange() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        load_graph({"schema_version": 1, "nodes": [], "edges": [], "Module": "truth"})


def test_rejects_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        load_graph(
            {
                "schema_version": 1,
                "nodes": [{"id": "a", "package": ""}],
                "edges": [{"source": "a", "target": "b", "relation": "Call"}],
            }
        )


def test_overlapping_partition_is_not_silently_scored() -> None:
    architecture = Architecture(
        components=[
            Component("one", "", ["a", "b"]),
            Component("two", "", ["b", "c"]),
        ]
    )
    with pytest.raises(ValueError, match="multiple components"):
        partition_labels(architecture, ["a", "b", "c"])


def test_missing_partition_entity_is_rejected() -> None:
    with pytest.raises(ValueError, match="Missing"):
        partition_labels(Architecture(components=[Component("one", "", ["a"])]), ["a", "b"])


def test_real_wca_preserves_complete_file_partition() -> None:
    graph = load_graph(
        {
            "schema_version": 1,
            "nodes": [{"id": name, "package": "pkg"} for name in ("a", "b", "c")],
            "edges": [{"source": "a", "target": "b", "relation": "Call"}],
        }
    )
    result = run(graph, 2)
    assert result["nodes"] == ["a", "b", "c"]
    assert result["results"]["wca"]["status"] == "ok"
    assert len(result["results"]["wca"]["labels"]) == 3
