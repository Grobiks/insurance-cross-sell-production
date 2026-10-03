"""Command line interface: `insurance download | train | tune | predict`."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from . import config
from .data import download_data
from .predict import predict_file
from .train import prepare_split, train_models
from .tuning import grid_search, optuna_search


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insurance", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download the raw dataset (cached)")
    download.add_argument("--data-path", type=Path, default=config.DATA_PATH)
    download.add_argument("--force", action="store_true")
    download.add_argument(
        "--workers", type=int, default=32, help="parallel connections (1 = single stream)"
    )

    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-path", type=Path, default=config.DATA_PATH)
        p.add_argument("--sample-frac", type=float, default=config.SAMPLE_FRAC)
        p.add_argument(
            "--nrows", type=int, default=None, help="read only the first N rows (smoke runs)"
        )

    choices = [*config.MODEL_NAMES, "all"]

    train = sub.add_parser("train", help="fit models with the stored best hyperparameters")
    add_data_args(train)
    train.add_argument("--model", choices=choices, default="all")
    train.add_argument("--out-dir", type=Path, default=config.ARTIFACTS_DIR)
    train.add_argument("--threshold", type=float, default=config.THRESHOLD)
    train.add_argument(
        "--params-file", type=Path, default=None, help="JSON with parameter overrides (from `tune`)"
    )

    tune = sub.add_parser("tune", help="re-run hyperparameter search (slow)")
    add_data_args(tune)
    tune.add_argument("--model", choices=config.MODEL_NAMES, required=True)
    tune.add_argument("--method", choices=["grid", "optuna"], default="optuna")
    tune.add_argument("--n-trials", type=int, default=30)
    tune.add_argument("--out-file", type=Path, default=None)

    predict = sub.add_parser("predict", help="score a CSV with a saved model")
    predict.add_argument("--model-path", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and args.params_file and args.model == "all":
        parser.error("--params-file holds parameters of one model; pass --model as well")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.command == "download":
        download_data(args.data_path, force=args.force, workers=args.workers)
    elif args.command == "train":
        names = config.MODEL_NAMES if args.model == "all" else [args.model]
        params = (
            json.loads(args.params_file.read_text(encoding="utf-8")) if args.params_file else None
        )
        table = train_models(
            names,
            args.data_path,
            args.sample_frac,
            args.out_dir,
            args.threshold,
            params,
            args.nrows,
        )
        print(table.to_string())
    elif args.command == "tune":
        split = prepare_split(args.data_path, args.sample_frac, args.nrows)
        if args.method == "grid":
            best, score = grid_search(args.model, split.X_train, split.y_train)
        else:
            best, score = optuna_search(args.model, split.X_train, split.y_train, args.n_trials)
        print(f"best CV F1={score:.4f}\n{json.dumps(best, indent=2)}")
        out_file = args.out_file or config.ARTIFACTS_DIR / f"best_params_{args.model}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(best, indent=2), encoding="utf-8")
    elif args.command == "predict":
        result = predict_file(args.model_path, args.input, args.output)
        print(f"Wrote {len(result)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
