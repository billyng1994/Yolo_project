"""Evaluate a trusted local detector on the original validation or test split."""

from __future__ import annotations

import json

from .config import cpu_threads, inference_parser, local_data, project_path, trusted_model
from .dataset import atomic_write
from .hardware import plan


def main(argv: list[str] | None = None) -> None:
    parser = inference_parser(__doc__ or "Evaluate detector", validation=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--plots", action="store_true")
    args = vars(parser.parse_args(argv))
    try:
        model_path = trusted_model(args.pop("model"))
        args["data"] = local_data(args["data"], (args["split"],))
        threads = args.pop("cpu_threads")
        args["project"] = project_path(args["project"])
        hardware = plan(device=args.pop("device"), batch=args["batch"],
                        workers=args.pop("workers"), training=False, amp=False)
        if args["imgsz"] < 32:
            raise ValueError("Image size must be >=32")
        with cpu_threads(threads) as callback:
            from ultralytics import YOLO

            model = YOLO(model_path)
            if callback is not None:
                model.add_callback("on_val_start", callback)
            metrics = model.val(device=hardware.device, workers=hardware.workers,
                                half=False, **args)
        results = {key: float(value) for key, value in metrics.results_dict.items()}
        atomic_write(metrics.save_dir / "metrics.json", json.dumps(results, indent=2, allow_nan=False).encode())
        print(json.dumps(results, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()