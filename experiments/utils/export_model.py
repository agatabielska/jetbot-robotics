"""Utilities for exporting trained models and related artifacts.

Main helper:
	export_best_model_to_onnx(...)

It exports a model to ONNX and stores additional artifacts next to it:
  - test transforms (JSON)
  - best-epoch metadata such as loss/accuracy (JSON)
"""

from __future__ import annotations

from datetime import datetime
import inspect
import json
from pathlib import Path
from typing import Any

import torch


def _detect_caller_stem() -> str:
	"""Return stem of file that called the export function."""
	this_file = Path(__file__).resolve()
	for frame in inspect.stack()[2:]:
		frame_path = Path(frame.filename).resolve()
		if frame_path != this_file:
			return frame_path.stem
	return "model"


def _to_json_compatible(value: Any) -> Any:
	"""Convert objects (including torchvision transforms) to JSON-compatible data."""
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(k): _to_json_compatible(v) for k, v in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_to_json_compatible(v) for v in value]

	# Handle torch/tensor values commonly found in metadata.
	if isinstance(value, torch.Tensor):
		return value.detach().cpu().tolist()

	# For transform objects and most classes, serialize public attributes.
	if hasattr(value, "__dict__"):
		attrs = {
			k: _to_json_compatible(v)
			for k, v in vars(value).items()
			if not k.startswith("_")
		}
		return {
			"type": value.__class__.__name__,
			"module": value.__class__.__module__,
			"attrs": attrs,
		}

	return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=True)


def export_best_model_to_onnx(
	model: torch.nn.Module,
	sample_input: torch.Tensor | tuple[torch.Tensor, ...],
	test_transform: Any | None = None,
	best_epoch_metadata: dict[str, Any] | None = None,
	output_root: str | Path = "bestmodels",
	caller_file: str | Path | None = None,
	opset_version: int = 17,
	dynamic_batch: bool = True,
) -> dict[str, str]:
	"""Export a model to ONNX and save transform + metadata side artifacts.

	Output layout:
	  - bestmodels/<caller>_<timestamp>.onnx
	  - bestmodels/<caller>_<timestamp>/test_transform.json
	  - bestmodels/<caller>_<timestamp>/best_epoch_metadata.json

	Args:
		model: Trained PyTorch model instance.
		sample_input: Dummy input tensor (or tuple of tensors) for ONNX tracing.
		test_transform: Transform object used for evaluation/test dataset.
		best_epoch_metadata: Metrics/details for the best epoch, e.g. loss/accuracy.
		output_root: Root directory where artifacts are saved.
		caller_file: Optional explicit caller file; if omitted, detected automatically.
		opset_version: ONNX opset version.
		dynamic_batch: If True, ONNX first dimension is dynamic batch size.

	Returns:
		Dict with absolute paths for ONNX and sidecar JSON files.
	"""
	if caller_file is None:
		caller_stem = _detect_caller_stem()
	else:
		caller_stem = Path(caller_file).stem

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	base_name = f"{caller_stem}_{timestamp}"

	output_root = Path(output_root)
	output_root.mkdir(parents=True, exist_ok=True)

	onnx_path = output_root / f"{base_name}.onnx"
	sidecar_dir = output_root / base_name
	sidecar_dir.mkdir(parents=True, exist_ok=True)

	model.eval()

	if isinstance(sample_input, tuple):
		input_names = [f"input_{idx}" for idx in range(len(sample_input))]
		export_input = sample_input
	else:
		input_names = ["input"]
		export_input = sample_input

	with torch.no_grad():
		model_out = model(*export_input) if isinstance(export_input, tuple) else model(export_input)

	if isinstance(model_out, (tuple, list)):
		output_names = [f"output_{idx}" for idx in range(len(model_out))]
	else:
		output_names = ["output"]

	dynamic_axes: dict[str, dict[int, str]] | None = None
	if dynamic_batch:
		dynamic_axes = {name: {0: "batch"} for name in input_names}
		for name in output_names:
			dynamic_axes[name] = {0: "batch"}

	torch.onnx.export(
		model,
		export_input,
		f=onnx_path,
		export_params=True,
		do_constant_folding=True,
		opset_version=opset_version,
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
	)

	transform_payload = {
		"saved_at": datetime.now().isoformat(timespec="seconds"),
		"test_transform": _to_json_compatible(test_transform),
	}
	_write_json(sidecar_dir / "test_transform.json", transform_payload)

	metadata_payload = {
		"saved_at": datetime.now().isoformat(timespec="seconds"),
		"best_epoch_metadata": _to_json_compatible(best_epoch_metadata or {}),
	}
	_write_json(sidecar_dir / "best_epoch_metadata.json", metadata_payload)

	return {
		"onnx_path": str(onnx_path.resolve()),
		"transform_path": str((sidecar_dir / "test_transform.json").resolve()),
		"metadata_path": str((sidecar_dir / "best_epoch_metadata.json").resolve()),
	}


def export_run_onnx(
	model: torch.nn.Module,
	sample_input: torch.Tensor | tuple[torch.Tensor, ...],
	run_name: str = "run1",
	role: str = "best",  # 'best' or 'last'
	test_transform: Any | None = None,
	metadata: dict[str, Any] | None = None,
	output_root: str | Path = "bestmodels",
	opset_version: int = 17,
	dynamic_batch: bool = True,
) -> dict[str, str]:
	"""Export ONNX inside a run folder and save role-specific sidecars.

	Files written:
	  - <output_root>/<run_name>/best.onnx  (if role=='best')
	  - <output_root>/<run_name>/last.onnx  (if role=='last')
	  - <output_root>/<run_name>/{role}_test_transform.json
	  - <output_root>/<run_name>/{role}_metadata.json
	"""
	run_dir = Path(output_root) / run_name
	run_dir.mkdir(parents=True, exist_ok=True)

	filename = "best.onnx" if role == "best" else "last.onnx"
	onnx_path = run_dir / filename

	model.eval()

	# prepare input and names (same logic as export_best_model_to_onnx)
	if isinstance(sample_input, tuple):
		input_names = [f"input_{i}" for i in range(len(sample_input))]
		export_input = sample_input
	else:
		input_names = ["input"]
		export_input = sample_input

	with torch.no_grad():
		model_out = model(*export_input) if isinstance(export_input, tuple) else model(export_input)

	if isinstance(model_out, (tuple, list)):
		output_names = [f"output_{i}" for i in range(len(model_out))]
	else:
		output_names = ["output"]

	dynamic_axes = {name: {0: "batch"} for name in input_names} if dynamic_batch else None
	if dynamic_axes is not None:
		for name in output_names:
			dynamic_axes[name] = {0: "batch"}

	torch.onnx.export(
		model,
		export_input,
		f=onnx_path,
		export_params=True,
		do_constant_folding=True,
		opset_version=opset_version,
		input_names=input_names,
		output_names=output_names,
		dynamic_axes=dynamic_axes,
	)

	# sidecar files
	transform_payload = {
		"saved_at": datetime.now().isoformat(timespec="seconds"),
		"test_transform": _to_json_compatible(test_transform),
	}
	_write_json(run_dir / f"{role}_test_transform.json", transform_payload)

	metadata_payload = {
		"saved_at": datetime.now().isoformat(timespec="seconds"),
		"metadata": _to_json_compatible(metadata or {}),
	}
	_write_json(run_dir / f"{role}_metadata.json", metadata_payload)

	return {
		"onnx_path": str(onnx_path.resolve()),
		"transform_path": str((run_dir / f"{role}_test_transform.json").resolve()),
		"metadata_path": str((run_dir / f"{role}_metadata.json").resolve()),
	}
