"""Compatibility reader for immutable paper-v1.0 archives.

Active code/configuration/output uses paper names. This module translates only
archive paths and metadata; observations, queries and weight tensors are unchanged.
The original file hashes and provenance remain in manifests/archive/paper-v1.0/.
"""
import csv
import io
import json
import re
from pathlib import PurePosixPath

_RENAMES = (
    ("CLIQUEestimator_bound_mono", "BOLTEstimator"),
    ("CLIQUEestimator_bound", "UBLBEstimator"),
    ("CLIQUEdataset_bound", "UBLBDataset"),
    ("CLIQUEestimator_mono", "UBOrderEstimator"),
    ("CLIQUE_bound_mono", "BOLT"),
    ("CLIQUE_bound", "UB_LB"),
    ("CLIQUE_mono", "UB_ORDER"),
    ("CLIQUE-BOUND-MONO-STRICT-LAMBDA0", "BOLT_L0"),
    ("CLIQUE-BOUND-MONO-LAMBDA0", "UB_LB"),
    ("CLIQUE-BOUND-MONO", "BOLT"),
    ("CLIQUE-BOUND", "UB_LB"),
    ("CLIQUE-MONO", "UB_ORDER"),
    ("BIMO-L0", "BOLT_L0"),
    ("BIMO", "BOLT"),
    ("bimo", "bolt"),
    ("MID_MONO", "MID_ORDER"),
    ("MID-MONO", "MID-ORDER"),
    ("NO_BOUND_MONO", "NO_BOUND_ORDER"),
    ("NoBoundMono", "NoBoundOrder"),
    ("no_bound_mono", "no_bound_order"),
    ("SKIP_LEGACY_MONO_EVAL", "SKIP_LEGACY_ORDER_EVAL"),
    ("lambda_mono", "lambda_order"),
    ("mono_tolerance", "order_tolerance"),
    ("train_mono", "train_order"),
)

def paper_names(text):
    for old, new in _RENAMES:
        text = text.replace(old, new)
    text = re.sub(r"(?<![A-Za-z0-9])MONO(?![A-Za-z0-9])", "UB_ORDER", text)
    text = re.sub(r"(?<!NO_)(?<!NO-)(?<!UB_)(?<![A-Za-z0-9])BOUND(?![A-Za-z0-9])", "UB_LB", text)
    return text

# These values are observations, not method identifiers or provenance paths.
_OBSERVATIONS = {"query", "queries", "pattern", "patterns", "sql", "true", "est",
                 "tight_query", "loose_query", "test_patterns", "chain_index"}

def _metadata(value, field=""):
    if field in _OBSERVATIONS:
        return value
    if isinstance(value, dict):
        return {paper_names(k): _metadata(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_metadata(v, field) for v in value]
    return paper_names(value) if isinstance(value, str) else value


def materialized_bytes(path, data):
    """Rename metadata only. Numeric prediction files and workloads stay byte exact."""
    if not path.startswith(("artifacts/checkpoints/", "artifacts/timing/", "artifacts/job/")):
        return data
    suffix = PurePosixPath(path).suffix
    if suffix not in (".json", ".jsonl", ".yml", ".yaml", ".csv"):
        return data
    text = data.decode("utf-8")
    if paper_names(text) == text:
        return data
    if suffix == ".json":
        value = json.loads(text)
        # The nested keys of positive-prediction maps are actual LIKE predicates.
        if PurePosixPath(path).name == "positive_predictions.json":
            value = {paper_names(k): v for k, v in value.items()}
        else:
            value = _metadata(value)
        return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    if suffix == ".jsonl":
        return "".join(json.dumps(_metadata(json.loads(line)), ensure_ascii=False) + "\n"
                       for line in text.splitlines() if line.strip()).encode()
    if suffix in (".yml", ".yaml"):
        import yaml
        return yaml.safe_dump(_metadata(yaml.safe_load(text)), sort_keys=True).encode()
    reader = csv.DictReader(io.StringIO(text))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=[paper_names(k) for k in reader.fieldnames])
    writer.writeheader()
    writer.writerows(_metadata(row) for row in reader)
    return output.getvalue().encode()
