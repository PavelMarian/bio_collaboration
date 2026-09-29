from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cv_module.config import load_config
from cv_module.inference.backend import ConfigurableBackend
from cv_module.inference.pipeline import AnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Parser configured for analysis and model-validation commands.
    """
    parser = argparse.ArgumentParser(description="Analyze activated-sludge microscope images")
    parser.add_argument("source", type=Path, nargs="?", help="JPG/BMP file or directory")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--floc-model", type=Path, help="Override floc segmentation weights")
    parser.add_argument("--microorganism-model", type=Path, help="Override microorganism detector weights")
    parser.add_argument("--device", help="Override device: cpu, 0, 1, ...")
    parser.add_argument("--validate-models", action="store_true", help="Load weights and verify required class names")
    return parser


def main() -> None:
    """Load configuration, validate models, and run the requested analysis.

    Raises:
        SystemExit: If model classes are incompatible or the source is missing.
    """
    args = build_parser().parse_args()
    config = load_config(args.config)
    overrides = {}
    if args.floc_model:
        overrides["floc_model"] = args.floc_model.expanduser().resolve()
    if args.microorganism_model:
        overrides["microorganism_model"] = args.microorganism_model.expanduser().resolve()
    if args.device:
        overrides["device"] = args.device
    config = replace(config, **overrides)
    backend = ConfigurableBackend(config)
    pipeline = AnalysisPipeline(config, backend)
    class_errors = pipeline.validate_model_classes()
    if class_errors:
        raise SystemExit("Model validation failed:\n- " + "\n- ".join(class_errors))
    if args.validate_models:
        print("Models are readable and contain all configured classes.")
        return
    if args.source is None:
        raise SystemExit("source is required unless --validate-models is used")
    summary = pipeline.run(args.source, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
