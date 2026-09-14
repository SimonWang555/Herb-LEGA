from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .bases import BaseDataset


@dataclass(frozen=True)
class HerbRecord:
    global_image_idx: int
    category: str
    raw_class_id: int
    class_index: int
    image_name: str
    image_path: str
    descriptions: Tuple[str, str, str]


def _first_present(mapping: Mapping, keys: Sequence[str], default=None):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def _normalize_descriptions(item: Mapping) -> Tuple[str, str, str]:
    value = _first_present(
        item,
        ("descriptions", "captions", "texts", "sentences", "description"),
    )
    if isinstance(value, dict):
        ordered = []
        for key in ("S1", "S2", "S3", "s1", "s2", "s3", "1", "2", "3"):
            if key in value:
                ordered.append(value[key])
        value = ordered
    if isinstance(value, str):
        separate = [
            _first_present(item, ("description_1", "desc1", "s1", "S1")),
            _first_present(item, ("description_2", "desc2", "s2", "S2")),
            _first_present(item, ("description_3", "desc3", "s3", "S3")),
        ]
        value = separate if all(isinstance(x, str) and x.strip() for x in separate) else [value]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            "Every Herb163CMR image must have exactly three descriptions; "
            f"got {value!r}."
        )
    result = tuple(str(x).strip() for x in value)
    if any(not x for x in result):
        raise ValueError("Empty Herb163CMR description found.")
    return result  # type: ignore[return-value]


def _resolve_image_path(
    dataset_root: Path,
    category: str,
    image_name: str,
    explicit_path,
) -> Path:
    candidates: List[Path] = []
    if explicit_path not in (None, ""):
        raw = Path(str(explicit_path))
        candidates.append(raw if raw.is_absolute() else dataset_root / raw)
        if dataset_root.name in raw.parts:
            anchor = raw.parts.index(dataset_root.name)
            candidates.append(dataset_root / Path(*raw.parts[anchor + 1 :]))
        candidates.append(dataset_root / category / raw.name)
        candidates.append(dataset_root / raw.name)
    candidates.extend([dataset_root / category / image_name, dataset_root / image_name])

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized not in seen:
            unique.append(candidate)
            seen.add(normalized)
    for candidate in unique:
        if candidate.is_file():
            return candidate.resolve()
    attempted = "\n  - ".join(str(path) for path in unique)
    raise FileNotFoundError(f"Image file not found. Tried:\n  - {attempted}")


def _read_jsonl(path: Path, require_object: bool = True) -> List:
    rows: List = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if require_object and not isinstance(item, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}.")
            rows.append(item)
    return rows


def load_herb_records(
    dataset_root: Union[str, Path], source_jsonl: Union[str, Path]
) -> List[HerbRecord]:
    dataset_root = Path(dataset_root)
    source_jsonl = Path(source_jsonl)
    if not source_jsonl.is_file():
        raise FileNotFoundError(f"Source JSONL does not exist: {source_jsonl}")
    rows = _read_jsonl(source_jsonl)
    if not rows:
        raise ValueError(f"No records found in {source_jsonl}")

    categories: List[str] = []
    raw_ids: List[Optional[int]] = []
    for row in rows:
        category = _first_present(
            row,
            ("category", "class_name", "label_name", "herb_name", "class"),
        )
        if category is None:
            raise KeyError("Missing category/class_name/herb_name field.")
        categories.append(str(category))
        raw_id = _first_present(row, ("class_id", "category_id", "label_id"))
        raw_ids.append(int(raw_id) if raw_id is not None else None)

    fallback = {name: idx for idx, name in enumerate(sorted(set(categories)))}
    resolved_raw_ids = [fallback[c] if rid is None else rid for c, rid in zip(categories, raw_ids)]
    dense_map = {raw_id: idx for idx, raw_id in enumerate(sorted(set(resolved_raw_ids)))}

    records: List[HerbRecord] = []
    seen_global_ids = set()
    seen_keys = set()
    for line_idx, (row, category, raw_class_id) in enumerate(
        zip(rows, categories, resolved_raw_ids)
    ):
        global_id = int(
            _first_present(
                row,
                ("global_image_idx", "global_idx", "image_idx", "index", "idx"),
                line_idx,
            )
        )
        if global_id in seen_global_ids:
            raise ValueError(f"Duplicate global_image_idx: {global_id}")
        seen_global_ids.add(global_id)

        image_name = _first_present(
            row,
            ("image_name", "filename", "file_name", "image", "img_name"),
        )
        path_hint = _first_present(row, ("image_path", "path", "img_path"))
        if image_name is None:
            if path_hint is None:
                raise KeyError("Missing both image_name and image_path.")
            image_name = Path(str(path_hint)).name
        image_name = Path(str(image_name)).name
        key = (category, image_name)
        if key in seen_keys:
            raise ValueError(f"Duplicate category/image_name pair: {key}")
        seen_keys.add(key)

        image_path = _resolve_image_path(dataset_root, category, image_name, path_hint)
        records.append(
            HerbRecord(
                global_image_idx=global_id,
                category=category,
                raw_class_id=int(raw_class_id),
                class_index=dense_map[int(raw_class_id)],
                image_name=image_name,
                image_path=str(image_path),
                descriptions=_normalize_descriptions(row),
            )
        )

    records.sort(key=lambda record: record.global_image_idx)
    if len(records) != 20000:
        raise ValueError(
            f"Herb163CMR must contain exactly 20,000 images; got {len(records):,}."
        )
    class_count = len({record.class_index for record in records})
    if class_count != 163:
        raise ValueError(
            f"Herb163CMR must contain exactly 163 classes; got {class_count}."
        )
    return records


def _build_lookups(records: Sequence[HerbRecord]):
    global_to_record = {record.global_image_idx: record for record in records}
    key_to_global: Dict[Tuple[str, str], int] = {}
    path_to_global: Dict[str, int] = {}
    basename_to_globals: Dict[str, List[int]] = {}
    for record in records:
        gid = record.global_image_idx
        key_to_global[(record.category, record.image_name)] = gid
        path_candidates = {
            str(Path(record.image_path)),
            str(Path(record.category) / record.image_name),
            record.image_name,
        }
        for candidate in path_candidates:
            path_to_global[candidate] = gid
            basename_to_globals.setdefault(Path(candidate).name, []).append(gid)
    for basename, values in basename_to_globals.items():
        basename_to_globals[basename] = sorted(set(values))
    return global_to_record, key_to_global, path_to_global, basename_to_globals


def _entry_to_global_id(
    entry,
    global_to_record,
    key_to_global,
    path_to_global,
    basename_to_globals,
) -> int:
    if isinstance(entry, (int, np.integer)):
        value = int(entry)
        if value not in global_to_record:
            raise KeyError(f"Unknown global image index: {value}")
        return value
    if isinstance(entry, str):
        text = entry.strip()
        if text.lstrip("-").isdigit():
            return _entry_to_global_id(
                int(text), global_to_record, key_to_global, path_to_global, basename_to_globals
            )
        normalized = str(Path(text))
        if normalized in path_to_global:
            return path_to_global[normalized]
        hits = basename_to_globals.get(Path(text).name, [])
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"Cannot resolve split entry string: {entry}")
    if isinstance(entry, dict):
        index = _first_present(
            entry,
            ("global_image_idx", "global_idx", "image_idx", "index", "idx"),
        )
        if index is not None:
            return _entry_to_global_id(
                int(index), global_to_record, key_to_global, path_to_global, basename_to_globals
            )
        category = _first_present(
            entry,
            ("category", "class_name", "label_name", "herb_name", "class"),
        )
        image_name = _first_present(
            entry,
            ("image_name", "filename", "file_name", "image", "img_name"),
        )
        if category is not None and image_name is not None:
            key = (str(category), Path(str(image_name)).name)
            if key in key_to_global:
                return key_to_global[key]
        path_hint = _first_present(entry, ("image_path", "path", "img_path"))
        if path_hint is not None:
            path = Path(str(path_hint))
            if str(path) in path_to_global:
                return path_to_global[str(path)]
            if category is not None:
                key = (str(category), path.name)
                if key in key_to_global:
                    return key_to_global[key]
            hits = basename_to_globals.get(path.name, [])
            if len(hits) == 1:
                return hits[0]
        if image_name is not None:
            hits = basename_to_globals.get(Path(str(image_name)).name, [])
            if len(hits) == 1:
                return hits[0]
    raise TypeError(f"Unsupported split entry: {entry!r}")


def load_canonical_splits(
    split_dir: Union[str, Path],
    records: Sequence[HerbRecord],
    strict: bool = True,
    expected_counts: Optional[Tuple[int, int, int]] = (15995, 2028, 1977),
) -> Dict[str, List[int]]:
    split_dir = Path(split_dir)
    paths = {
        "train": split_dir / "train.jsonl",
        "val": split_dir / "validation.jsonl",
        "test": split_dir / "test.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The shared canonical split files are required and will not be regenerated. Missing:\n"
            + "\n".join(missing)
        )

    global_to_record, key_to_global, path_to_global, basename_to_globals = _build_lookups(records)
    result: Dict[str, List[int]] = {}
    for split, path in paths.items():
        result[split] = [
            _entry_to_global_id(
                row,
                global_to_record,
                key_to_global,
                path_to_global,
                basename_to_globals,
            )
            for row in _read_jsonl(path, require_object=False)
        ]

    sets = {split: set(values) for split, values in result.items()}
    for split, values in result.items():
        if len(values) != len(sets[split]):
            raise ValueError(f"Duplicate image in {split} split.")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Overlap detected between train/validation/test splits.")
    all_globals = set(global_to_record)
    union = sets["train"] | sets["val"] | sets["test"]
    if union != all_globals:
        missing_ids = sorted(all_globals - union)
        extra_ids = sorted(union - all_globals)
        raise ValueError(
            "Canonical split union does not exactly cover the source dataset. "
            f"missing={missing_ids[:10]}, extra={extra_ids[:10]}"
        )

    expected_classes = {record.class_index for record in records}
    for split, values in result.items():
        present = {global_to_record[value].class_index for value in values}
        if strict and present != expected_classes:
            raise ValueError(
                f"{split} contains {len(present)} classes; expected {len(expected_classes)}."
            )

    actual_counts = (len(result["train"]), len(result["val"]), len(result["test"]))
    if expected_counts is not None and actual_counts != expected_counts:
        raise ValueError(
            "This package is configured for the fixed Herb163CMR seed_42 split. "
            f"actual train/val/test={actual_counts}, expected={expected_counts}."
        )

    print(
        "Using canonical Herb163CMR image-level splits: "
        f"train={actual_counts[0]:,}, val={actual_counts[1]:,}, test={actual_counts[2]:,}."
    )
    return {name: sorted(values) for name, values in result.items()}


class Herb163CMR(BaseDataset):
    """Herb163CMR reader for the shared image-level seed-42 split."""

    def __init__(
        self,
        root: str,
        source_jsonl: str,
        split_dir: str,
        strict: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.dataset_root = Path(root)
        self.source_jsonl = Path(source_jsonl)
        self.split_dir = Path(split_dir)
        self.records = load_herb_records(self.dataset_root, self.source_jsonl)
        self.record_by_global = {record.global_image_idx: record for record in self.records}
        self.splits = load_canonical_splits(self.split_dir, self.records, strict=strict)

        self.train_annos = [self.record_by_global[idx] for idx in self.splits["train"]]
        self.val_annos = [self.record_by_global[idx] for idx in self.splits["val"]]
        self.test_annos = [self.record_by_global[idx] for idx in self.splits["test"]]

        self.train, self.train_id_container = self._process_train(self.train_annos)
        self.val, self.val_id_container = self._process_eval(self.val_annos)
        self.test, self.test_id_container = self._process_eval(self.test_annos)

        if verbose:
            self.logger.info("=> Herb163CMR images and descriptions are loaded")
            self.show_dataset_info()

    @staticmethod
    def _process_train(records: Sequence[HerbRecord]):
        dataset = []
        ids = set()
        for record in records:
            ids.add(record.class_index)
            for caption_idx, caption in enumerate(record.descriptions):
                dataset.append(
                    (
                        record.class_index,
                        record.global_image_idx,
                        record.image_path,
                        caption,
                        caption_idx,
                    )
                )
        return dataset, ids

    @staticmethod
    def _process_eval(records: Sequence[HerbRecord]):
        img_paths: List[str] = []
        image_pids: List[int] = []
        image_ids: List[int] = []
        captions: List[str] = []
        caption_pids: List[int] = []
        caption_image_ids: List[int] = []
        caption_indices: List[int] = []
        ids = set()
        for record in records:
            ids.add(record.class_index)
            img_paths.append(record.image_path)
            image_pids.append(record.class_index)
            image_ids.append(record.global_image_idx)
            for caption_idx, caption in enumerate(record.descriptions):
                captions.append(caption)
                caption_pids.append(record.class_index)
                caption_image_ids.append(record.global_image_idx)
                caption_indices.append(caption_idx)
        dataset = {
            "img_paths": img_paths,
            "image_pids": image_pids,
            "image_ids": image_ids,
            "captions": captions,
            "caption_pids": caption_pids,
            "caption_image_ids": caption_image_ids,
            "caption_indices": caption_indices,
        }
        return dataset, ids
