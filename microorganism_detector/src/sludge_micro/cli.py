"""Command-line interface ``sludge-micro``.

Commands that use a delivered model take ``--passport`` and ``--weights`` and need no
project configuration; the other commands read the configuration given by ``--config``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from sludge_micro.config import ProjectConfig, load_config
from sludge_micro.messages import ProjectError, text
from sludge_micro.runtime import network_disabled

LOG = logging.getLogger("sludge_micro")


def cmd_audit(config: ProjectConfig, _: argparse.Namespace) -> str:
    """Audit raw archives."""
    from sludge_micro.audit import run_audit

    run_audit(config)
    return text("audit_done", path=config.paths.experiments_dir / "data_audit.json")


def cmd_prepare(config: ProjectConfig, _: argparse.Namespace) -> str:
    """Prepare the microorganism set from the audited inventory."""
    from sludge_micro.prepare import run_prepare

    report = run_prepare(config)
    excluded = report["images_total"] - report["images_included"]
    return text("prepare_done", included=report["images_included"], excluded=excluded)


def cmd_variants(config: ProjectConfig, _: argparse.Namespace) -> str:
    """Build research variants for conflicting copies."""
    from sludge_micro.variants import run_variants

    document = run_variants(config)
    return text(
        "variants_done",
        count=len(document["variants"]),
        matches=document["working_policy_matches_prepare"],
    )


def cmd_split(config: ProjectConfig, _: argparse.Namespace) -> str:
    """Build and lock the temporal split."""
    from sludge_micro.split import lock_path, run_split

    run_split(config)
    return text("split_done", path=lock_path(config).relative_to(config.root))


def cmd_propose_split(config: ProjectConfig, _: argparse.Namespace) -> str:
    """Suggest temporal boundaries from confirmed capture days."""
    from sludge_micro.reporting import write_json
    from sludge_micro.split import propose_boundaries

    path = config.resolve(config.paths.experiments_dir) / "split_proposal.json"
    write_json(path, propose_boundaries(config))
    return text("proposal_done", path=path.relative_to(config.root))


def cmd_preflight(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Check hardware, weights and one training step before a run."""
    from sludge_micro.reporting import write_json
    from sludge_micro.train import run_preflight

    report = run_preflight(config)
    # Named after the configuration file, so configurations of one architecture keep their own.
    name = f"preflight_{args.config.stem}.json"
    path = config.resolve(config.paths.outputs_dir) / name
    write_json(path, report)
    return text("preflight_done", path=path.relative_to(config.root))


def cmd_train(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Train one run and evaluate it on validation."""
    from sludge_micro.runs import checkpoint_of
    from sludge_micro.train import run_train

    run_train(config, args.run_id, args.resume)
    path = checkpoint_of(config, args.run_id).relative_to(config.root)
    return text("train_done", run_id=args.run_id, path=path)


def cmd_evaluate(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Evaluate a run on validation or, once, on test."""
    from sludge_micro.runs import evaluate_run

    record = evaluate_run(config, args.run_id, args.split)
    return text("evaluate_done", run_id=args.run_id, split=args.split, macro_f1=record["macro_f1"])


def run_predictor(config: ProjectConfig, run_id: str) -> object:
    """Load a run with the confidence selected on validation."""
    from sludge_micro.reporting import read_json
    from sludge_micro.runs import run_dirs
    from sludge_micro.runs import run_predictor as load

    metrics = run_dirs(config, run_id)[0] / "metrics.json"
    if not metrics.is_file():
        raise ProjectError("confidence_unselected", run_id=run_id)
    return load(config, run_id, read_json(metrics)["confidence"])


def cmd_inspect(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Describe a checkpoint without modifying it."""
    from sludge_micro.checkpoint import inspect_checkpoint
    from sludge_micro.reporting import write_json

    write_json(args.output, inspect_checkpoint(args.path, config.model.architecture))
    return text("inspect_done", path=args.output)


def cmd_compare(config: ProjectConfig, _: argparse.Namespace) -> str:
    """Compare validation results of the runs in the experiments folder."""
    from sludge_micro.compare import run_comparison

    document = run_comparison(config)
    return text(
        "compare_done", count=len(document["runs"]), selected=document["decision"]["selected"]
    )


def cmd_examples(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Render validation examples of correct detections, misses and false positives."""
    from sludge_micro.examples import run_examples

    report = run_examples(config, args.run_id, args.limit)
    return text("examples_done", run_id=args.run_id, count=len(report["examples"]))


def cmd_handover(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Check the export and the pipeline hand-over on validation frames."""
    from sludge_micro.handover import run_handover

    report = run_handover(config, args.run_id, args.limit)
    return text(
        "handover_done",
        run_id=args.run_id,
        images=report["images"],
        matches=report["detections_match"],
    )


def cmd_predict(config: ProjectConfig | None, args: argparse.Namespace) -> str:
    """Predict detections and counts with a delivered model or a run of the configuration."""
    from sludge_micro.infer import discover_images, predict_images, write_outputs

    if args.passport is not None:
        from sludge_micro.passport import load_passport, passport_predictor

        passport = load_passport(args.passport)
        predictor = passport_predictor(passport, args.weights)
        suffixes, classes = passport.image.formats, passport.names
    else:
        predictor = run_predictor(config, required_run(args))
        suffixes, classes = config.audit.image_suffixes, config.classes.names
    items = discover_images(args.input, suffixes)
    predictions = predict_images(predictor, items)
    write_outputs(predictions, classes, args.output)
    return text("predict_done", path=args.output, images=len(predictions))


def cmd_export_pavel(config: ProjectConfig | None, args: argparse.Namespace) -> str:
    """Write weights, categories and configuration block for the Pavel pipeline."""
    if args.passport is not None:
        from sludge_micro.adapter import export_passport
        from sludge_micro.passport import load_passport

        export_passport(load_passport(args.passport), args.weights, args.output)
        return text("export_done", path=args.output)
    from sludge_micro.adapter import export_for_pavel
    from sludge_micro.reporting import read_json
    from sludge_micro.runs import checkpoint_of, run_dirs

    run_id = required_run(args)
    confidence = read_json(run_dirs(config, run_id)[0] / "metrics.json")["confidence"]
    export_for_pavel(config, checkpoint_of(config, run_id), confidence, args.output)
    return text("export_done", path=args.output)


def cmd_passport(config: ProjectConfig, args: argparse.Namespace) -> str:
    """Write the passport of a trained run of the configuration."""
    from sludge_micro.delivery import build_passport
    from sludge_micro.reporting import write_json

    write_json(args.output, build_passport(config, required_run(args)))
    return text("passport_done", path=args.output)


def cmd_check_model(_: ProjectConfig | None, args: argparse.Namespace) -> str:
    """Check delivered weights against their passport and load them."""
    from sludge_micro.passport import load_passport, passport_model

    passport = load_passport(args.passport)
    passport_model(passport, args.weights)
    return text("model_ok", run_id=passport.run_id, sha256=passport.weights.sha256)


def cmd_check_pipeline(_: ProjectConfig | None, args: argparse.Namespace) -> str:
    """Compare the Pavel pipeline with own inference of a delivered model on a folder."""
    from sludge_micro.handover import run_passport_check
    from sludge_micro.passport import load_passport

    passport = load_passport(args.passport)
    report = run_passport_check(passport, args.weights, args.pipeline, args.input, args.report)
    return text(
        "pipeline_check_done",
        path=args.report,
        images=report["images"],
        matches=report["detections_match"],
        counts=report["counts_match_detections"],
    )


def cmd_passport_readme(_: ProjectConfig | None, args: argparse.Namespace) -> str:
    """Rewrite the passport block of a README from the passport."""
    from sludge_micro.delivery import update_readme
    from sludge_micro.passport import load_passport

    changed = update_readme(args.readme, load_passport(args.passport))
    return text("readme_done", path=args.readme, changed=changed)


def required_run(args: argparse.Namespace) -> str:
    """The run id of a configuration-based command.

    Raises:
        ProjectError: If no run id was given.
    """
    if not args.run_id:
        raise ProjectError("run_id_missing", command=args.command)
    return args.run_id


Command = Callable[[ProjectConfig | None, argparse.Namespace], str]
COMMANDS: dict[str, tuple[str, Command]] = {
    "audit": ("Аудит исходных архивов", cmd_audit),
    "prepare": ("Подготовка набора микроорганизмов", cmd_prepare),
    "variants": ("Варианты обработки копий с расходящейся разметкой", cmd_variants),
    "split": ("Разбиение выборок по протоколу конфигурации", cmd_split),
    "propose-split": ("Предложение временных границ", cmd_propose_split),
    "preflight": ("Проверка оборудования, весов и шага обучения", cmd_preflight),
    "train": ("Обучение запуска", cmd_train),
    "evaluate": ("Оценка запуска", cmd_evaluate),
    "compare": ("Сравнение запусков на validation", cmd_compare),
    "examples": ("Примеры детекций, пропусков и ложных срабатываний", cmd_examples),
    "handover": ("Проверка передачи запуска в пайплайн Павла", cmd_handover),
    "passport": ("Паспорт обученного запуска", cmd_passport),
    "predict": ("Предсказание для файла или каталога", cmd_predict),
    "export-pavel": ("Экспорт для пайплайна Павла", cmd_export_pavel),
    "check-model": ("Проверка весов по паспорту", cmd_check_model),
    "check-pipeline": ("Проверка поставляемой модели в пайплайне Павла", cmd_check_pipeline),
    "passport-readme": ("Обновление блока паспорта в README", cmd_passport_readme),
    "inspect-checkpoint": ("Описание checkpoint", cmd_inspect),
}
# Commands that never read the project configuration.
PASSPORT_COMMANDS = {"check-model", "check-pipeline", "passport-readme"}

RUN_ID = ("--run-id", {"required": True, "help": "идентификатор запуска"})
OPTIONAL_RUN = ("--run-id", {"default": None, "help": "запуск конфигурации; без --passport"})
PASSPORT = ("--passport", {"type": Path, "help": "паспорт поставляемой модели"})
WEIGHTS = ("--weights", {"type": Path, "help": "веса поставляемой модели"})
NEEDED_PASSPORT = ("--passport", {"type": Path, "required": True, "help": "паспорт модели"})
NEEDED_WEIGHTS = ("--weights", {"type": Path, "required": True, "help": "веса модели"})
OUTPUT = ("--output", {"type": Path, "required": True})
INPUT = ("--input", {"type": Path, "required": True})
EXTRA_ARGS: dict[str, tuple[tuple[str, dict], ...]] = {
    "train": (RUN_ID, ("--resume", {"action": "store_true", "help": "продолжить после эпохи"})),
    "evaluate": (RUN_ID, ("--split", {"choices": ["validation", "test"], "required": True})),
    "compare": (),
    "examples": (RUN_ID, ("--limit", {"type": int, "default": 6})),
    "handover": (RUN_ID, ("--limit", {"type": int, "default": 10})),
    "passport": (RUN_ID, OUTPUT),
    "predict": (OPTIONAL_RUN, PASSPORT, WEIGHTS, INPUT, OUTPUT),
    "export-pavel": (OPTIONAL_RUN, PASSPORT, WEIGHTS, OUTPUT),
    "check-model": (NEEDED_PASSPORT, NEEDED_WEIGHTS),
    "check-pipeline": (
        NEEDED_PASSPORT,
        NEEDED_WEIGHTS,
        INPUT,
        ("--pipeline", {"type": Path, "required": True, "help": "каталог с cv_module"}),
        ("--report", {"type": Path, "required": True}),
    ),
    "passport-readme": (NEEDED_PASSPORT, ("--readme", {"type": Path, "required": True})),
    "inspect-checkpoint": (("--path", {"type": Path, "required": True}), OUTPUT),
}


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with one sub-command per operation."""
    parser = argparse.ArgumentParser(prog="sludge-micro", description="Детектор микроорганизмов")
    parser.add_argument("--verbose", action="store_true", help="подробный журнал")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _) in COMMANDS.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=Path("configs/microorganisms.yaml"))
        for flag, kwargs in EXTRA_ARGS.get(name, ()):
            command.add_argument(flag, **kwargs)
    return parser


def needs_config(args: argparse.Namespace) -> bool:
    """Whether a command reads the project configuration."""
    if args.command in PASSPORT_COMMANDS:
        return False
    return getattr(args, "passport", None) is None


def main(argv: list[str] | None = None) -> int:
    """Run one operation with network access disabled.

    Returns:
        Process exit code: 0 on success, 2 on a reported project error.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        with network_disabled():
            config = load_config(args.config) if needs_config(args) else None
            if getattr(args, "passport", None) is not None and getattr(args, "weights", "") is None:
                raise ProjectError("weights_missing", command=args.command)
            message = COMMANDS[args.command][1](config, args)
    except ProjectError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
