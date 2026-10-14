import json

import pytest

from kaggle_gpu_mcp.service import Service


def test_dataset_registry_records_local_provenance(settings):
    file = settings.workspace / "observations.csv"
    file.write_text("x,y,label\n1,2,0\n2,,1\n")
    service = Service(settings)
    with pytest.raises(ValueError):
        service.register_source("local_csv", "observations.csv", "CC0-1.0")
    result = service.register_source("local_csv", "observations.csv", "CC0-1.0", approved=True)
    assert result["metadata"]["row_count"] == 2
    assert result["metadata"]["missing_values"]["y"] == 1
    assert result["metadata"]["numeric_columns"] == ["x", "y", "label"]
    assert service.db.datasets()[0]["id"] == result["id"]


def test_kaggle_reference_requires_explicit_license(settings):
    service = Service(settings)
    with pytest.raises(ValueError):
        service.register_source("kaggle", "invalid-ref", "CC0-1.0", approved=True)
    result = service.register_source("kaggle", "team/dataset", "CC0-1.0", approved=True)
    assert result["metadata"]["checked_remote"] is False


def test_csv_merge_provenance_and_atomic_error(settings):
    service = Service(settings)
    (settings.workspace / "a.csv").write_text("f1,f2\n1,2\n2,3\n")
    (settings.workspace / "b.csv").write_text("f1,f2\n2,3\n3,4\n")
    merged = service.merge_csv(["a.csv", "b.csv"], "experiment.csv")
    report = json.loads((settings.workspace / merged["provenance_manifest"]).read_text())
    assert report["rows"] == 3
    assert merged["sha256"] == report["output_sha256"]
    assert len(report["input_sources"]) == 2
    assert not list((settings.workspace / "datasets" / "merged").glob(".merge-*"))
    (settings.workspace / "broken.csv").write_text("f1,f2\n1,2,3\n")
    with pytest.raises(ValueError):
        service.merge_csv(["a.csv", "broken.csv"], "never.csv")
    assert not (settings.workspace / "datasets" / "merged" / "never.csv").exists()


def test_csv_profile_rejects_duplicate_headers(settings):
    source = settings.workspace / "duplicate.csv"
    source.write_text("a,a\n1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        Service(settings).csv_profile("duplicate.csv")
