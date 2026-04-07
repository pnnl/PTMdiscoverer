"""Command-line interface for PTMdiscoverer.

This CLI exposes the core standalone workflow without requiring Docker,
MCP, or the ADEPT app stack.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .inference import (
    assess_study_folder,
    run_and_aggregate_multiple_inferences,
    run_inference,
    validate_study_folder,
)
from .reporting import analyze_run_results
from .utils import extract_delta_masses, load_protein_metadata


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _describe_run(args: argparse.Namespace, model: str) -> str:
    run_word = "run" if args.n_runs == 1 else "runs"
    return (
        f"Starting PTM inference for study folder '{args.study_folder}' with model '{model}' "
        f"and {args.n_runs} {run_word}. Runtime depends on the model, prompt size, and number of runs..."
    )


def _default_model() -> str:
    return os.environ.get("OPENAI_MODEL") or "gpt-4o"


def _list_proteins(study_folder: str) -> dict[str, Any]:
    validate_study_folder(study_folder)
    metadata_path = os.path.join(study_folder, "protein_metadata.json")
    proteins = load_protein_metadata(metadata_path)

    result: list[dict[str, Any]] = []
    for protein in proteins:
        delta_mass_file = protein.get("delta_mass_file", "")
        if delta_mass_file and not os.path.isabs(delta_mass_file):
            delta_mass_file = os.path.join(study_folder, delta_mass_file)
        result.append(
            {
                "uniprot_id": protein.get("uniprot_id", ""),
                "name": protein.get("name", ""),
                "has_delta_masses": bool(delta_mass_file and os.path.isfile(delta_mass_file)),
            }
        )
    return {"study_folder": study_folder, "count": len(result), "proteins": result}


def _extract_delta_masses_for_protein(
    study_folder: str,
    uniprot_id: str,
    *,
    max_preview_rows: int,
) -> dict[str, Any]:
    validate_study_folder(study_folder)

    metadata_path = os.path.join(study_folder, "protein_metadata.json")
    proteins = load_protein_metadata(metadata_path)
    protein = next(
        (entry for entry in proteins if str(entry.get("uniprot_id", "")).upper() == uniprot_id.upper()),
        None,
    )
    if protein is None:
        raise FileNotFoundError(f"Protein '{uniprot_id}' not found in metadata.")

    delta_mass_file = protein.get("delta_mass_file", "")
    if delta_mass_file and not os.path.isabs(delta_mass_file):
        delta_mass_file = os.path.join(study_folder, delta_mass_file)

    if delta_mass_file and os.path.isfile(delta_mass_file):
        frame = pd.read_csv(delta_mass_file)
        return {
            "uniprot_id": uniprot_id,
            "source": "existing_csv",
            "file": delta_mass_file,
            "rows": len(frame),
            "columns": list(frame.columns),
            "preview": frame.head(max_preview_rows).to_dict(orient="records"),
        }

    labels_path = os.path.join(study_folder, "dataset_labels.csv")
    msrun_table = pd.read_csv(labels_path)
    extracted = extract_delta_masses(uniprot_id, msrun_table, study_folder)
    if extracted.empty:
        return {
            "uniprot_id": uniprot_id,
            "source": "extraction",
            "rows": 0,
            "message": "No qualifying delta masses found in PSM data.",
        }

    out_path = os.path.join(study_folder, f"{uniprot_id}_delta-masses.csv")
    extracted.to_csv(out_path, index=False)
    return {
        "uniprot_id": uniprot_id,
        "source": "extraction",
        "file": out_path,
        "rows": len(extracted),
        "columns": list(extracted.columns),
        "preview": extracted.head(max_preview_rows).to_dict(orient="records"),
    }


def _get_run_results(
    *,
    study_folder: str | None,
    run_dir: str | None,
    max_preview_records: int,
) -> dict[str, Any]:
    if run_dir is None and study_folder is None:
        raise ValueError("Provide either 'study_folder' or 'run_dir'.")

    if study_folder is not None:
        validate_study_folder(study_folder)

    if run_dir is None:
        assert study_folder is not None
        output_dir = os.path.join(study_folder, "output")
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"No output directory found in '{study_folder}'.")
        runs = sorted(
            name
            for name in os.listdir(output_dir)
            if name.startswith("output_") and os.path.isdir(os.path.join(output_dir, name))
        )
        return {"study_folder": study_folder, "available_runs": runs}

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")

    predictions: list[dict[str, Any]] = []
    for json_path in sorted(glob.glob(os.path.join(run_dir, "ptm_predictions_*.json"))):
        basename = os.path.basename(json_path)
        stem = os.path.splitext(basename)[0]
        parts = stem.replace("ptm_predictions_", "").rsplit("_", 1)
        protein_id = parts[0] if parts else "unknown"
        with open(json_path, encoding="utf-8") as handle:
            records = json.load(handle)
        predictions.append(
            {
                "protein_id": protein_id,
                "file": json_path,
                "total_records": len(records),
                "preview": records[:max_preview_records],
            }
        )
    return {"run_dir": run_dir, "predictions": predictions}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptmdiscoverer",
        description=(
            "Run PTMdiscoverer without Docker, MCP, or the ADEPT app. "
            "For inference commands, set OPENAI_API_KEY and optionally OPENAI_BASE_URL in your shell. "
            "Use --model on the run command to choose the inference model."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a study folder and return structured feedback.",
    )
    validate_parser.add_argument("study_folder", help="Path to the study folder.")

    list_parser = subparsers.add_parser("list", help="List proteins in a study folder.")
    list_parser.add_argument("study_folder", help="Path to the study folder.")

    extract_parser = subparsers.add_parser(
        "extract",
        help="Read or extract delta masses for a single protein.",
    )
    extract_parser.add_argument("study_folder", help="Path to the study folder.")
    extract_parser.add_argument("uniprot_id", help="Target UniProt accession.")
    extract_parser.add_argument(
        "--max-preview-rows",
        type=int,
        default=10,
        help="Number of delta-mass rows to include in preview output.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run PTM inference once or multiple times with aggregation.",
    )
    run_parser.add_argument("study_folder", help="Path to the study folder.")
    run_parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of inference runs. Values > 1 trigger aggregation.",
    )
    run_parser.add_argument(
        "--model",
        default="",
        help=(
            "LLM model name for inference. Defaults to OPENAI_MODEL if set, otherwise gpt-4o. "
            "Model names are provider-specific."
        ),
    )
    run_parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL. Overrides OPENAI_BASE_URL for this command.",
    )
    run_parser.add_argument(
        "--aggregated-csv-path",
        default=None,
        help="Optional path for the aggregated CSV when n-runs > 1.",
    )

    results_parser = subparsers.add_parser(
        "results",
        help="List available runs or inspect prediction artifacts in a specific run directory.",
    )
    results_group = results_parser.add_mutually_exclusive_group(required=True)
    results_group.add_argument(
        "--study-folder",
        default=None,
        help="Study folder whose output_<timestamp> run directories should be listed.",
    )
    results_group.add_argument(
        "--run-dir",
        default=None,
        help="Specific output_<timestamp> run directory whose prediction JSON files should be previewed.",
    )
    results_parser.add_argument(
        "--max-preview-records",
        type=int,
        default=20,
        help="Number of prediction records per protein to include in preview output.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze per-protein prediction CSVs from one run, or a single aggregated predictions CSV.",
    )
    analyze_group = analyze_parser.add_mutually_exclusive_group(required=True)
    analyze_group.add_argument(
        "--run-dir",
        default=None,
        help="Specific output_<timestamp> run directory containing ptm_predictions_*.csv files.",
    )
    analyze_group.add_argument(
        "--aggregated-csv",
        default=None,
        help="Single aggregated_predictions_*.csv file to analyze.",
    )
    analyze_parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top recurring PTM calls/sites to include in summary tables and plots.",
    )
    analyze_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional parent directory for the generated analysis_<timestamp> report directory.",
    )
    analyze_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Generate JSON, Markdown, and CSV outputs only, without PNG plots.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            payload = assess_study_folder(args.study_folder)
        elif args.command == "list":
            payload = _list_proteins(args.study_folder)
        elif args.command == "extract":
            payload = _extract_delta_masses_for_protein(
                args.study_folder,
                args.uniprot_id,
                max_preview_rows=args.max_preview_rows,
            )
        elif args.command == "run":
            if args.n_runs < 1:
                raise ValueError("--n-runs must be >= 1.")
            model = args.model or _default_model()
            _print_status(_describe_run(args, model))
            if args.n_runs == 1:
                payload = run_inference(
                    base_data_folder=args.study_folder,
                    model=model,
                    base_url=args.base_url,
                )
                payload["n_runs"] = 1
                payload["model"] = model
            else:
                output_dir = os.path.join(args.study_folder, "output")
                existing_aggregated = set(glob.glob(os.path.join(output_dir, "aggregated_predictions_*.csv")))
                run_dirs, aggregated = run_and_aggregate_multiple_inferences(
                    args.n_runs,
                    base_data_folder=args.study_folder,
                    model=model,
                    base_url=args.base_url,
                    aggregated_csv_path=args.aggregated_csv_path,
                )
                aggregated_csv_path = args.aggregated_csv_path
                if aggregated_csv_path is None:
                    new_aggregated = sorted(
                        set(glob.glob(os.path.join(output_dir, "aggregated_predictions_*.csv"))) - existing_aggregated
                    )
                    aggregated_csv_path = new_aggregated[-1] if new_aggregated else None
                payload = {
                    "n_runs": args.n_runs,
                    "model": model,
                    "run_dirs": run_dirs,
                    "aggregated_rows": len(aggregated),
                    "aggregated_csv_path": aggregated_csv_path,
                }
        elif args.command == "results":
            payload = _get_run_results(
                study_folder=args.study_folder,
                run_dir=args.run_dir,
                max_preview_records=args.max_preview_records,
            )
        elif args.command == "analyze":
            payload = analyze_run_results(
                run_dir=args.run_dir,
                aggregated_csv=args.aggregated_csv,
                top_k=args.top_k,
                output_dir=args.output_dir,
                create_plots=not args.no_plots,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        _print_json({"error": str(exc)})
        return 1

    _print_json(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
