"""Merge tile NPZ shards into a single PLY file for inspection/rendering."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Optional
import sys


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "Grendel-GS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import torch
import utils.general_utils as utils
from src.tile_storage import TileStorage
from src.tile_training_utils import load_tiles_as_tensors
from scene.gaussian_model import GaussianModel



def _normalise_tile_id(tile_id: str) -> str:
	"""Allow callers to pass tile IDs with optional prefixes or extensions."""
	candidate = Path(tile_id).stem  # strips directories + extensions
	if candidate.startswith("tile_"):
		candidate = candidate[5:]
	return candidate



def _parse_args() -> Namespace:
	parser = ArgumentParser(description="Merge tile shards into a single PLY")
	parser.add_argument(
		"--tiles-root",
		required=True,
		help="Directory containing tiles_metadata.json and tiles/",
	)
	parser.add_argument(
		"--output-ply",
		required=True,
		help="Destination path for the merged PLY",
	)
	parser.add_argument(
		"--device",
		default="cuda:0",
		help="Torch device to stage tensors on (fallback to CPU if CUDA is unavailable)",
	)
	parser.add_argument(
		"--tile-ids",
		nargs="+",
		help=(
			"Optional list of tile ids (e.g. 0_10_0) to include. Defaults to all tiles found "
			"in tiles_metadata.json."
		),
	)
	return parser.parse_args()


def _make_dummy_args(output_dir: Path) -> Namespace:
	return Namespace(
		gaussians_distribution=False,
		distributed_save=False,
		model_path=str(output_dir),
		log_folder=str(output_dir),
		check_cpu_memory=False,
		check_gpu_memory=False,
		auto_start_checkpoint=False,
		quiet=True,
	)



def export_tiles_to_ply(
	*,
	tiles_root: Path,
	output_ply: Path,
	device: str = "cuda:0",
	tile_ids: Optional[Iterable[str]] = None,
) -> Path:
	tiles_root = tiles_root.resolve()
	output_ply = output_ply.resolve()
	output_ply.parent.mkdir(parents=True, exist_ok=True)

	if torch.cuda.is_available():
		device_t = torch.device(device)
	else:
		device_t = torch.device("cpu")

	storage = TileStorage(str(tiles_root))
	try:
		all_tiles = storage.metadata["tiles"]
		if tile_ids:
			normalised = [_normalise_tile_id(tid) for tid in tile_ids]
			missing = sorted(
				{
					orig
					for orig, norm in zip(tile_ids, normalised)
					if norm not in all_tiles
				}
			)
			if missing:
				raise KeyError(
					"Requested tile ids missing from metadata: " + ", ".join(missing)
				)

			tile_list = []
			seen = set()
			for norm in normalised:
				if norm not in seen:
					tile_list.append(norm)
					seen.add(norm)
		else:
			tile_list = sorted(all_tiles.keys())

		if not tile_list:
			raise RuntimeError(f"No tiles found under {tiles_root}")

		batch = load_tiles_as_tensors(storage, tile_list, device=device_t)
	finally:
		storage.close()

	sh_degree = batch.infer_sh_degree()
	model = GaussianModel(sh_degree)
	model.active_sh_degree = sh_degree
	batch.populate_gaussian_model(model)

	dummy_args = _make_dummy_args(output_ply.parent)
	utils.set_args(dummy_args)
	utils.DEFAULT_GROUP = utils.SingleGPUGroup()
	utils.IN_NODE_GROUP = utils.SingleGPUGroup()

	log_path = output_ply.with_suffix(".export.log")
	with log_path.open("w", encoding="utf-8") as log_file:
		utils.set_log_file(log_file)
		model.save_ply(str(output_ply))

	suffix = f"_rk{utils.GLOBAL_RANK}_ws{utils.WORLD_SIZE}"
	actual_path = output_ply.with_name(output_ply.stem + suffix + output_ply.suffix)
	if actual_path.exists() and actual_path != output_ply:
		actual_path.replace(output_ply)

	return output_ply


def main() -> None:
	args = _parse_args()
	export_tiles_to_ply(
		tiles_root=Path(args.tiles_root),
		output_ply=Path(args.output_ply),
		device=args.device,
		tile_ids=args.tile_ids,
	)


if __name__ == "__main__":
	main()