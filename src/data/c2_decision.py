import json
import hashlib
import os
import random
from pathlib import Path

DATASET_FILE = "c2_decision_v4.jsonl"


def _normalize_record(item):
    item = dict(item)
    if "shared_scenario" not in item and "scenario" in item:
        item["shared_scenario"] = item["scenario"]
    item.setdefault("commander_note", "")
    item.setdefault("intel_note", "")
    item.setdefault("safety_note", "")
    item.setdefault("task_type", "general_c2")
    item.setdefault("risk_option", None)
    item.setdefault("catastrophic_option", None)
    return item


def _candidate_paths(args):
    module_dir = Path(__file__).resolve().parent
    bundled_c2_dir = module_dir / "c2"
    project_root = module_dir.parent.parent
    cwd = Path.cwd()
    data_dir = getattr(args, "data_dir", None)
    explicit = getattr(args, "c2_data_file", None)
    version = getattr(args, "c2_version", "v4")
    if version != "v4":
        raise ValueError(f"Unsupported C2 dataset version: {version}")

    candidates = []

    if explicit:
        explicit_path = Path(explicit)
        candidates.extend([explicit_path, cwd / explicit_path])
        if explicit_path.name:
            candidates.extend([
                module_dir / explicit_path.name,
                bundled_c2_dir / explicit_path.name,
                project_root / explicit_path.name,
            ])
    else:
        if data_dir:
            data_dir_path = Path(data_dir)
            candidates.extend([
                data_dir_path / DATASET_FILE,
                data_dir_path / "c2" / DATASET_FILE,
                cwd / data_dir_path / DATASET_FILE,
                cwd / data_dir_path / "c2" / DATASET_FILE,
            ])

        for base in [
            bundled_c2_dir,
            module_dir,
            cwd / "src" / "data" / "c2",
            cwd / "src" / "data",
            project_root / "src" / "data" / "c2",
            project_root / "src" / "data",
        ]:
            candidates.append(base / DATASET_FILE)

    seen = set()
    result = []
    for cand in candidates:
        norm = os.path.normpath(str(cand))
        if norm not in seen:
            seen.add(norm)
            result.append(str(cand))
    return result


def load_data(args, split="test"):
    file_path = None
    checked = _candidate_paths(args)
    for cand in checked:
        if cand and os.path.exists(cand):
            file_path = cand
            break
    if file_path is None:
        raise FileNotFoundError("No C2 data file found. Checked: " + " | ".join(checked))

    setattr(args, "_c2_loaded_file", os.path.abspath(file_path))
    dataset_bytes = Path(file_path).read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    setattr(args, "_c2_dataset_sha256", dataset_sha256)

    metadata_path = Path(file_path).with_name("dataset_metadata.json")
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_file = metadata.get("dataset_file")
        expected_hash = str(metadata.get("dataset_sha256") or "").lower()
        if expected_file == Path(file_path).name and expected_hash and expected_hash != dataset_sha256:
            raise ValueError(
                f"Dataset checksum mismatch for {file_path}: expected {expected_hash}, got {dataset_sha256}."
            )
        setattr(args, "_c2_dataset_metadata_file", str(metadata_path.resolve()))

    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(_normalize_record(json.loads(line)))

    if metadata and metadata.get("dataset_file") == Path(file_path).name:
        expected_count = int(metadata.get("scenario_count", len(samples)))
        expected_types = int(metadata.get("scenario_type_count", len({row.get('task_type') for row in samples})))
        actual_types = len({row.get("task_type") for row in samples})
        if len(samples) != expected_count or actual_types != expected_types:
            raise ValueError(
                f"Dataset metadata mismatch: expected {expected_count} scenarios/{expected_types} types, "
                f"got {len(samples)} scenarios/{actual_types} types."
            )

    rng = random.Random(getattr(args, "seed", 42))
    rng.shuffle(samples)
    if getattr(args, "data_size", 0) and args.data_size > 0:
        samples = samples[:args.data_size]

    labels = [f"({item['label'].strip().upper()})" for item in samples]
    return samples, labels
