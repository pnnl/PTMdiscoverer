"""Inference workflow: annotate delta-mass tables using an LLM."""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TextIO

import openai
import pandas as pd

from ptmdiscoverer.utils import (
    augment_protein_positions,
    build_prompt,
    extract_delta_masses,
    load_protein_metadata,
    parse_first_json_block,
    validate_ptm_records,
)

logger = logging.getLogger(__name__)

# Monotonic counter to guarantee unique timestamps even in tight loops.
_ts_counter = 0
_ts_last = ""


def assess_study_folder(
    base_data_folder: str,
    *,
    base_prompt_filename: str = "base_prompt.txt",
    protein_metadata_filename: str = "protein_metadata.json",
    dataset_labels_filename: str = "dataset_labels.csv",
) -> dict[str, Any]:
    """Return a structured validation report for a PTM study folder."""
    errors: list[str] = []
    warnings: list[str] = []

    is_directory = os.path.isdir(base_data_folder)
    base_prompt_path = os.path.join(base_data_folder, base_prompt_filename)
    protein_metadata_path = os.path.join(base_data_folder, protein_metadata_filename)
    dataset_labels_path = os.path.join(base_data_folder, dataset_labels_filename)
    msfragger_path = os.path.join(base_data_folder, "Results_MSFragger")
    output_path = os.path.join(base_data_folder, "output")

    if not is_directory:
        errors.append(f"Study folder '{base_data_folder}' does not exist or is not a directory.")

    base_prompt_exists = os.path.isfile(base_prompt_path)
    protein_metadata_exists = os.path.isfile(protein_metadata_path)
    dataset_labels_exists = os.path.isfile(dataset_labels_path)
    msfragger_exists = os.path.isdir(msfragger_path)
    output_exists = os.path.isdir(output_path)

    if is_directory and not base_prompt_exists:
        errors.append(f"Required file '{base_prompt_filename}' not found in study folder '{base_data_folder}'.")
    if is_directory and not protein_metadata_exists:
        errors.append(f"Required file '{protein_metadata_filename}' not found in study folder '{base_data_folder}'.")
    if is_directory and dataset_labels_exists and not msfragger_exists:
        errors.append(
            f"'{dataset_labels_filename}' exists but 'Results_MSFragger/' directory is missing in '{base_data_folder}'."
        )
    if is_directory and not dataset_labels_exists:
        warnings.append(
            f"Optional file '{dataset_labels_filename}' not found. Missing delta-mass CSVs cannot be auto-extracted."
        )
    if is_directory and not output_exists:
        warnings.append("No output/ directory found yet. This is normal before the first inference run.")

    protein_count: int | None = None
    metadata_format_ok: bool | None = None
    if protein_metadata_exists:
        try:
            proteins = load_protein_metadata(protein_metadata_path)
            protein_count = len(proteins)
            metadata_format_ok = True
            if protein_count == 0:
                warnings.append("protein_metadata.json contains zero valid protein entries.")
        except Exception as exc:  # noqa: BLE001
            metadata_format_ok = False
            errors.append(f"Failed to parse '{protein_metadata_filename}': {exc}")

    dataset_labels_readable: bool | None = None
    msrun_count: int | None = None
    if dataset_labels_exists:
        try:
            msrun_table = pd.read_csv(dataset_labels_path)
            dataset_labels_readable = True
            if "MSRUNID" not in msrun_table.columns:
                errors.append(f"'{dataset_labels_filename}' is missing required column 'MSRUNID'.")
            else:
                msrun_count = int(msrun_table["MSRUNID"].notna().sum())
                if msrun_count == 0:
                    warnings.append(f"'{dataset_labels_filename}' contains no MSRUNID rows.")
        except Exception as exc:  # noqa: BLE001
            dataset_labels_readable = False
            errors.append(f"Failed to read '{dataset_labels_filename}': {exc}")

    return {
        "study_folder": base_data_folder,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "protein_count": protein_count,
        "msrun_count": msrun_count,
        "checks": {
            "study_folder_exists": is_directory,
            "base_prompt_file": {"path": base_prompt_path, "exists": base_prompt_exists},
            "protein_metadata_file": {
                "path": protein_metadata_path,
                "exists": protein_metadata_exists,
                "parse_ok": metadata_format_ok,
            },
            "dataset_labels_file": {
                "path": dataset_labels_path,
                "exists": dataset_labels_exists,
                "readable": dataset_labels_readable,
            },
            "results_msfragger_dir": {
                "path": msfragger_path,
                "required": dataset_labels_exists,
                "exists": msfragger_exists,
            },
            "output_dir": {"path": output_path, "exists": output_exists},
        },
    }


def validate_study_folder(
    base_data_folder: str,
    *,
    base_prompt_filename: str = "base_prompt.txt",
    protein_metadata_filename: str = "protein_metadata.json",
    dataset_labels_filename: str = "dataset_labels.csv",
) -> None:
    """Validate that a study folder has all required inputs.

    Raises FileNotFoundError with a descriptive message on the first
    missing item.
    """
    report = assess_study_folder(
        base_data_folder,
        base_prompt_filename=base_prompt_filename,
        protein_metadata_filename=protein_metadata_filename,
        dataset_labels_filename=dataset_labels_filename,
    )
    if report["errors"]:
        raise FileNotFoundError(str(report["errors"][0]))


def _write_prediction_files(records: list[dict[str, Any]], pred_base: str) -> list[dict[str, Any]]:
    """Filter records to valid positions and write JSON + CSV prediction files.

    Returns the list of kept (valid) records. Does NOT mutate the input list.
    """
    import csv as _csv

    kept: list[dict[str, Any]] = []
    invalid_count = 0
    minus1_count = 0
    for rec in records:
        raw = rec.get("PTM.Protein.Position", -1)
        try:
            val = int(raw)
        except (ValueError, TypeError):  # int() can raise either for bad input
            val = -1
        if val > 0:
            kept.append(rec)
        else:
            invalid_count += 1
            if val == -1:
                minus1_count += 1
    if invalid_count:
        msg = f"Dropped {invalid_count} record(s) with invalid PTM.Protein.Position"
        if minus1_count:
            if minus1_count == invalid_count:
                msg += " (all were -1)"
            else:
                msg += f" ({minus1_count} were -1)"
        msg += f". Kept {len(kept)}."
        logger.warning(msg)
    if not kept:
        return []

    json_path = pred_base + ".json"
    csv_path = pred_base + ".csv"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(kept, jf, indent=2)
    fieldnames = sorted({k for r in kept for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = _csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return kept


def _write_html(prompt: str, answer: str, path: str) -> None:
    """Write simple HTML trace file with prompt and answer."""
    with open(path, "w", encoding="utf-8") as html_file:
        html_file.write(
            "\n--------- QUESTION ----------- \n"
            + '<pre style="background-color: lightblue;">\n'
            + prompt
            + "</pre>"
            + "\n--------- LLM OUTPUT ----------- \n"
            + '<pre style="background-color: lightgray;">\n'
            + answer
            + "</pre>"
        )


@dataclass
class InferenceContext:
    """Shared context for a single inference run.

    Bundles the parameters that ``process_protein`` needs so callers
    pass one typed object instead of many positional strings.
    """

    client: Any
    base_prompt: str
    model: str
    timestamp: str
    base_data_folder: str
    run_dir: str
    log_handle: TextIO
    msrun_table: pd.DataFrame


def process_protein(
    ctx: InferenceContext,
    protein: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    """Process one protein: optionally extract delta masses, invoke LLM, persist artifacts."""
    uniprot_id: str | None = protein.get("uniprot_id")
    name: str = str(protein.get("name", uniprot_id) or "")
    delta_mass_file: str | None = protein.get("delta_mass_file") or None
    _t_protein_start = time.monotonic()
    logger.info("Processing protein %s", uniprot_id)

    # Resolve relative paths (stored as relative to base_data_folder for portability)
    if delta_mass_file and not os.path.isabs(delta_mass_file):
        delta_mass_file = os.path.join(ctx.base_data_folder, delta_mass_file)

    metadata_changed_local = False

    needs_extraction = uniprot_id and (not delta_mass_file or not os.path.exists(delta_mass_file))
    if needs_extraction:
        assert uniprot_id is not None  # narrowed by `needs_extraction` guard above
        if ctx.msrun_table.empty:
            logger.warning("Cannot extract delta masses for %s; dataset labels file missing. Skipping.", uniprot_id)
            return False, []
        extracted_df = extract_delta_masses(uniprot_id, ctx.msrun_table, ctx.base_data_folder)
        if extracted_df.empty:
            logger.warning("No qualifying delta masses extracted for %s; skipping.", uniprot_id)
            return False, []
        delta_mass_file = f"{ctx.base_data_folder}/{uniprot_id}_delta-masses.csv"
        extracted_df.to_csv(delta_mass_file, index=False)
        # Store as relative path for portability
        protein["delta_mass_file"] = os.path.relpath(delta_mass_file, ctx.base_data_folder)
        metadata_changed_local = True
        logger.info("Extracted %d delta mass rows for %s -> %s", len(extracted_df), uniprot_id, delta_mass_file)

    if not (uniprot_id and delta_mass_file and os.path.exists(delta_mass_file)):
        logger.warning("Skipping %s; delta mass file unavailable after extraction attempt.", uniprot_id)
        return metadata_changed_local, []

    try:
        delta_mass_df = pd.read_csv(delta_mass_file)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):  # noqa: BLE001
        return metadata_changed_local, []

    prompt = build_prompt(ctx.base_prompt, name, delta_mass_df)
    logger.info(
        "Calling LLM for %s with model %s (prompt_chars=%d)",
        uniprot_id,
        ctx.model,
        len(prompt),
    )
    _t_llm = time.monotonic()
    # Use streaming to keep the connection alive and avoid gateway timeouts (504)
    # that occur when the full response takes longer than the gateway's idle timeout.
    chunks: list[str] = []
    _first_chunk_logged = False
    with ctx.client.chat.completions.create(
        model=ctx.model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for _chunk in stream:
            delta = _chunk.choices[0].delta.content or ""
            if delta:
                if not _first_chunk_logged:
                    logger.info(
                        "Received first response chunk for %s after %.1fs",
                        uniprot_id,
                        time.monotonic() - _t_llm,
                    )
                    _first_chunk_logged = True
                chunks.append(delta)
    answer = "".join(chunks)
    logger.info(
        "LLM stream complete for %s in %.1fs (answer_chars=%d)",
        uniprot_id,
        time.monotonic() - _t_llm,
        len(answer),
    )

    # Log prompt + raw answer
    ctx.log_handle.write(f"----- {uniprot_id} PROMPT -----\n{prompt}\n----- {uniprot_id} RESPONSE -----\n{answer}\n\n")

    # Parse JSON block
    records = parse_first_json_block(answer)
    # Attempt to augment missing / invalid positions using extraction table data
    if records:
        try:
            augment_protein_positions(records, delta_mass_df)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed deriving PTM.Protein.Position for %s: %s", uniprot_id, exc)
        records = validate_ptm_records(records)
    os.makedirs(ctx.run_dir, exist_ok=True)
    if records:
        pred_base = os.path.join(ctx.run_dir, f"ptm_predictions_{uniprot_id}_{ctx.timestamp}")
        records = _write_prediction_files(records, pred_base)
        logger.info("Wrote %d PTM predictions for %s -> %s.json/.csv", len(records), uniprot_id, pred_base)
    else:
        logger.info("No parsable JSON block for %s; no prediction files written.", uniprot_id)

    _write_html(prompt, answer, os.path.join(ctx.run_dir, f"output_{uniprot_id}_{ctx.timestamp}.html"))

    logger.info(
        "Finished protein %s with %d records in %.1fs",
        uniprot_id,
        len(records),
        time.monotonic() - _t_protein_start,
    )
    return metadata_changed_local, records


def format_timestamp(dt: datetime | None = None) -> str:
    """Return a unique timestamp string YYYY-MM-DD-HH-MM-SS-mmm.

    Uses a module-level monotonic counter to guarantee uniqueness even
    when called in a tight loop within the same millisecond.
    """
    global _ts_counter, _ts_last  # noqa: PLW0603
    dt = dt or datetime.now()
    ts = dt.strftime("%Y-%m-%d-%H-%M-%S-") + f"{int(dt.microsecond / 1000):03d}"
    if ts == _ts_last:
        _ts_counter += 1
        ts = f"{ts}-{_ts_counter}"
    else:
        _ts_last = ts
        _ts_counter = 0
    return ts


def run_inference(
    base_data_folder: str = ".",
    *,
    model: str = "gpt-4o",
    client: Any | None = None,
    timestamp: str | None = None,
    dataset_labels_filename: str = "dataset_labels.csv",
    base_prompt_filename: str = "base_prompt.txt",
    protein_metadata_filename: str = "protein_metadata.json",
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run inference over all proteins.

    Assumptions:
        All required input files (`dataset_labels_filename`, `base_prompt_filename`,
        `protein_metadata_filename`) reside directly within `base_data_folder`.

    Parameters:
        base_data_folder: Root folder containing inputs and where outputs will be created.
        model: LLM model name.
        client: Optional pre-configured OpenAI-compatible client (constructed if None).
        timestamp: Optional fixed timestamp (for tests); generated if None.
        dataset_labels_filename: Filename (not path) of CSV listing MS run IDs.
        base_prompt_filename: Filename of base prompt text file.
        protein_metadata_filename: Filename of protein metadata JSON file.
        base_url: Base URL for the OpenAI-compatible API endpoint.
    """
    validate_study_folder(
        base_data_folder,
        base_prompt_filename=base_prompt_filename,
        protein_metadata_filename=protein_metadata_filename,
        dataset_labels_filename=dataset_labels_filename,
    )

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY environment variable is required.")
        effective_base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = openai.OpenAI(api_key=api_key, base_url=effective_base_url)

    base_prompt_file = os.path.join(base_data_folder, base_prompt_filename)
    protein_metadata_file = os.path.join(base_data_folder, protein_metadata_filename)
    dataset_labels_file = os.path.join(base_data_folder, dataset_labels_filename)

    with open(base_prompt_file, encoding="utf-8") as handle:
        base_prompt = handle.read()

    proteins = load_protein_metadata(protein_metadata_file)
    try:
        msrun_table = pd.read_csv(dataset_labels_file)
    except FileNotFoundError:
        msrun_table = pd.DataFrame()
    if timestamp is None:
        timestamp = format_timestamp()
    run_dir = os.path.join(base_data_folder, "output", f"output_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"output_log_{timestamp}.txt")

    metadata_changed_global = False
    total_records = 0
    logger.info("Starting inference across %d proteins", len(proteins))
    with open(log_path, "w", encoding="utf-8") as log_handle:
        ctx = InferenceContext(
            client=client,
            base_prompt=base_prompt,
            model=model,
            timestamp=timestamp,
            base_data_folder=base_data_folder,
            run_dir=run_dir,
            log_handle=log_handle,
            msrun_table=msrun_table,
        )
        for i, protein in enumerate(proteins):
            uid = protein.get("uniprot_id", f"protein_{i}")
            logger.info("Protein %d/%d (%s): starting", i + 1, len(proteins), uid)
            _t0 = time.monotonic()
            changed, records = process_protein(ctx, protein)
            logger.info(
                "Protein %d/%d (%s): finished in %.1fs with %d records",
                i + 1,
                len(proteins),
                uid,
                time.monotonic() - _t0,
                len(records),
            )
            metadata_changed_global = metadata_changed_global or changed
            total_records += len(records)

    # Persist updated metadata if any changes occurred
    if metadata_changed_global:
        backup_path = protein_metadata_file + ".bak"
        try:
            if os.path.exists(protein_metadata_file):
                os.replace(protein_metadata_file, backup_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not create metadata backup: %s", exc)
        try:
            with open(protein_metadata_file, "w", encoding="utf-8") as md_handle:
                json.dump({"version": 1, "proteins": proteins}, md_handle, indent=4)
            logger.info("Updated protein metadata written to %s", protein_metadata_file)
        except Exception as exc:  # pragma: no cover
            logger.error("Error writing updated metadata: %s", exc)

    logger.info(
        "DONE! Processed %d proteins; total parsed PTM records: %d. Artifacts in run directory: %s",
        len(proteins),
        total_records,
        run_dir,
    )

    return {
        "run_dir": run_dir,
        "timestamp": timestamp,
        "proteins_processed": len(proteins),
        "total_ptm_records": total_records,
        "metadata_updated": metadata_changed_global,
    }


def run_and_aggregate_multiple_inferences(
    n: int = 10,
    *,
    base_data_folder: str = ".",
    model: str = "gpt-4o",
    base_url: str | None = None,
    dataset_labels_filename: str = "dataset_labels.csv",
    base_prompt_filename: str = "base_prompt.txt",
    protein_metadata_filename: str = "protein_metadata.json",
    client: Any | None = None,
    aggregated_csv_path: str | None = None,
    add_source_path: bool = True,
    add_run_index: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """Run inference `n` times and aggregate all prediction CSVs (mandatory).

    Returns:
        (run_dirs, aggregated_dataframe)

    Aggregated DataFrame provenance columns:
        * `Source.File`  - absolute path to originating per-protein CSV (if `add_source_path`)
        * `Run.Index`    - integer index corresponding to execution order (if `add_run_index`)
        * `Protein.ID`   - parsed from per-protein prediction filename
    """
    run_dirs: list[str] = []
    # Use a root timestamp for the multi-run batch (distinct from per-run timestamps)
    batch_timestamp = format_timestamp()
    for _ in range(n):
        ts = format_timestamp()
        run_inference(
            base_data_folder=base_data_folder,
            model=model,
            client=client,
            timestamp=ts,
            dataset_labels_filename=dataset_labels_filename,
            base_prompt_filename=base_prompt_filename,
            protein_metadata_filename=protein_metadata_filename,
            base_url=base_url,
        )
        run_dirs.append(os.path.join(base_data_folder, "output", f"output_{ts}"))

    frames: list[pd.DataFrame] = []
    for idx, rd in enumerate(run_dirs):
        pattern = os.path.join(rd, "ptm_predictions_*.csv")
        for csv_path in glob.glob(pattern):
            try:
                df = pd.read_csv(csv_path)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            if add_source_path:
                df["Source.File"] = os.path.abspath(csv_path)
            if add_run_index:
                df["Run.Index"] = idx
            base = os.path.basename(csv_path)
            if base.startswith("ptm_predictions_"):
                no_prefix = base[len("ptm_predictions_") :]
                stem = os.path.splitext(no_prefix)[0]
                parts = stem.split("_")
                if len(parts) >= 2:
                    df["Protein.ID"] = "_".join(parts[:-1])
            frames.append(df)
    aggregated = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if aggregated_csv_path is None:
        # Default path inside base_data_folder/output with batch timestamp
        os.makedirs(os.path.join(base_data_folder, "output"), exist_ok=True)
        aggregated_csv_path = os.path.join(
            base_data_folder,
            "output",
            f"aggregated_predictions_{batch_timestamp}.csv",
        )
    try:
        aggregated.to_csv(aggregated_csv_path, index=False)
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed writing aggregated CSV %s: %s", aggregated_csv_path, exc)
    return run_dirs, aggregated


def main() -> None:  # keep original entry point name
    """CLI entry point for running inference with default settings."""
    run_inference()


if __name__ == "__main__":
    main()
