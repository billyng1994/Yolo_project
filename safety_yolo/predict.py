"""Stream detection over images, directories, videos, or a numeric camera index."""

from __future__ import annotations

from pathlib import Path

from .config import cpu_threads, inference_parser, project_path, trusted_model
from .hardware import plan


def main(argv: list[str] | None = None) -> None:
    parser = inference_parser(__doc__ or "Stream predictions")
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = vars(parser.parse_args(argv))
    try:
        model_path = trusted_model(args.pop("model"))
        threads = args.pop("cpu_threads")
        args["project"] = project_path(args["project"])
        hardware = plan(device=args.pop("device"), batch=args["batch"],
                        workers=args.pop("workers"), training=False, amp=False)
        source = args.pop("source")
        source = int(source) if source.isascii() and source.isdecimal() else source
        if isinstance(source, str) and not Path(source).exists():
            raise ValueError("Source must be a local image, directory, video, or numeric camera index")
        if args["imgsz"] < 32 or not 0 <= args["conf"] <= 1 or not 0 <= args["iou"] <= 1:
            raise ValueError("Use imgsz>=32 and conf/iou between 0 and 1")
        save = not args.pop("no_save")
        count = 0
        with cpu_threads(threads) as callback:
            from ultralytics import YOLO

            model = YOLO(model_path)
            if callback is not None:
                model.add_callback("on_predict_start", callback)
            for _ in model.predict(source=source, stream=True, save=save,
                                   device=hardware.device, **args):
                count += 1
        print(f"Processed {count} images/frames")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()