"""Deterministic run-results analysis and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .inference import format_timestamp

REQUIRED_COLUMNS = {
    "PEPTIDE",
    "PTM.NAME",
    "PTM.Protein.Position",
    "Protein.ID",
    "Run.Index",
}
HIGH_CONSENSUS_THRESHOLD = 50.0
LOW_CONSENSUS_THRESHOLD = 20.0
PROTEIN_FOCUS_THRESHOLD = 50.0


def _parse_protein_id_from_prediction_path(prediction_path: Path) -> str:
    stem = prediction_path.stem.replace("ptm_predictions_", "")
    parts = stem.rsplit("_", 1)
    return parts[0] if parts else "unknown"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resolve_analysis_input(
    *,
    run_dir: str | None = None,
    aggregated_csv: str | None = None,
) -> dict[str, Any]:
    """Resolve and validate the requested analysis input."""
    if bool(run_dir) == bool(aggregated_csv):
        raise ValueError("Provide exactly one of 'run_dir' or 'aggregated_csv'.")

    if run_dir is not None:
        run_path = Path(run_dir)
        if not run_path.is_dir():
            raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")
        input_files = [str(path) for path in sorted(run_path.glob("ptm_predictions_*.csv"))]
        return {
            "analysis_mode": "run_dir",
            "run_dir": run_path,
            "aggregated_csv": None,
            "input_files": input_files,
            "output_parent": run_path,
        }

    assert aggregated_csv is not None
    csv_path = Path(aggregated_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Aggregated CSV '{aggregated_csv}' does not exist.")
    return {
        "analysis_mode": "aggregated_csv",
        "run_dir": None,
        "aggregated_csv": csv_path,
        "input_files": [str(csv_path)],
        "output_parent": csv_path.parent,
    }


def load_run_predictions(run_dir: str | Path) -> pd.DataFrame:
    """Load prediction CSV files from one run directory into a normalized dataframe."""
    run_path = Path(run_dir)
    frames: list[pd.DataFrame] = []
    prediction_files: list[Path] = sorted(run_path.glob("ptm_predictions_*.csv"))
    if not prediction_files:
        raise ValueError(f"No prediction CSV files found in run directory '{run_dir}'.")

    for csv_path in prediction_files:
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        protein_id = _parse_protein_id_from_prediction_path(csv_path)
        normalized = frame.copy()
        if "Protein.ID" not in normalized.columns:
            normalized["Protein.ID"] = protein_id
        else:
            normalized["Protein.ID"] = normalized["Protein.ID"].fillna(protein_id)
        if "Run.Index" not in normalized.columns:
            normalized["Run.Index"] = 0
        else:
            normalized["Run.Index"] = normalized["Run.Index"].fillna(0)
        if "Source.File" not in normalized.columns:
            normalized["Source.File"] = str(csv_path)
        else:
            normalized["Source.File"] = normalized["Source.File"].fillna(str(csv_path))
        frames.append(normalized)

    if not frames:
        raise ValueError(f"Prediction CSV files in run directory '{run_dir}' contain no rows.")

    return pd.concat(frames, ignore_index=True)


def load_aggregated_predictions(aggregated_csv: str | Path) -> pd.DataFrame:
    """Load an aggregated PTM predictions CSV."""
    frame = pd.read_csv(aggregated_csv)
    if frame.empty:
        raise ValueError(f"Aggregated CSV '{aggregated_csv}' contains no rows.")
    return frame


def _normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for analysis: {sorted(missing)}")

    normalized = frame.copy()
    normalized["PTM.Protein.Position"] = pd.to_numeric(normalized["PTM.Protein.Position"], errors="raise").astype(int)
    normalized["Run.Index"] = pd.to_numeric(normalized["Run.Index"], errors="raise").astype(int)
    normalized["PEPTIDE"] = normalized["PEPTIDE"].astype(str)
    normalized["PTM.NAME"] = normalized["PTM.NAME"].astype(str)
    normalized["Protein.ID"] = normalized["Protein.ID"].astype(str)
    if "PTM.DeltaMass" in normalized.columns:
        normalized["PTM.DeltaMass"] = pd.to_numeric(normalized["PTM.DeltaMass"], errors="coerce")
    return normalized


def _empty_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def build_summary_tables(frame: pd.DataFrame, *, top_k: int) -> dict[str, pd.DataFrame]:
    """Build deterministic summary tables used by the report and plots."""
    total_runs = max(int(frame["Run.Index"].nunique()), 1)

    ptm_frequency_by_site = (
        frame.groupby(["Protein.ID", "PEPTIDE", "PTM.NAME", "PTM.Protein.Position"], dropna=False)["Run.Index"]
        .nunique()
        .reset_index(name="RunCount")
        .sort_values(["RunCount", "Protein.ID", "PEPTIDE", "PTM.NAME"], ascending=[False, True, True, True])
    )
    ptm_frequency_by_site["PercentRuns"] = (ptm_frequency_by_site["RunCount"] / total_runs) * 100.0

    top_ptm_frames: list[pd.DataFrame] = []
    for _, subframe in ptm_frequency_by_site.groupby("Protein.ID", sort=True):
        ranked = subframe.head(top_k).copy()
        ranked.insert(0, "RankWithinProtein", range(1, len(ranked) + 1))
        top_ptm_frames.append(ranked)
    top_ptms_per_protein = pd.concat(top_ptm_frames, ignore_index=True) if top_ptm_frames else pd.DataFrame()

    peptide_position_frequency = (
        frame.groupby(["Protein.ID", "PEPTIDE", "PTM.Protein.Position"], dropna=False)["Run.Index"]
        .nunique()
        .reset_index(name="RunCount")
        .sort_values(["RunCount", "Protein.ID", "PEPTIDE"], ascending=[False, True, True])
    )
    peptide_position_frequency["PercentRuns"] = (peptide_position_frequency["RunCount"] / total_runs) * 100.0

    site_ptm_composition = (
        frame.groupby(["Protein.ID", "PEPTIDE", "PTM.Protein.Position", "PTM.NAME"], dropna=False)["Run.Index"]
        .nunique()
        .reset_index(name="RunCount")
        .sort_values(["RunCount", "Protein.ID", "PEPTIDE", "PTM.NAME"], ascending=[False, True, True, True])
    )
    site_ptm_composition["PercentRuns"] = (site_ptm_composition["RunCount"] / total_runs) * 100.0

    protein_prediction_counts = frame.groupby("Protein.ID", dropna=False).size().rename("TotalPredictions")
    protein_unique_sites = (
        frame[["Protein.ID", "PEPTIDE", "PTM.Protein.Position"]]
        .drop_duplicates()
        .groupby("Protein.ID", dropna=False)
        .size()
        .rename("UniqueSites")
    )
    protein_unique_ptms = frame.groupby("Protein.ID", dropna=False)["PTM.NAME"].nunique().rename("UniquePTMNames")

    top_site_by_protein = (
        peptide_position_frequency.sort_values(
            ["Protein.ID", "RunCount", "PEPTIDE", "PTM.Protein.Position"],
            ascending=[True, False, True, True],
        )
        .groupby("Protein.ID", as_index=False)
        .first()
        .rename(
            columns={
                "PEPTIDE": "TopSitePeptide",
                "PTM.Protein.Position": "TopSitePosition",
                "RunCount": "TopSiteRunCount",
                "PercentRuns": "TopSitePercentRuns",
            }
        )
    )
    top_site_prediction_fraction = (
        frame.groupby(["Protein.ID", "PEPTIDE", "PTM.Protein.Position"], dropna=False)
        .size()
        .rename("SitePredictionCount")
        .reset_index()
        .sort_values(
            ["Protein.ID", "SitePredictionCount", "PEPTIDE", "PTM.Protein.Position"],
            ascending=[True, False, True, True],
        )
        .groupby("Protein.ID", as_index=False)
        .first()
    )
    top_site_prediction_fraction = top_site_prediction_fraction.merge(
        protein_prediction_counts.reset_index(),
        on="Protein.ID",
        how="left",
    )
    top_site_prediction_fraction["TopSitePredictionFraction"] = (
        top_site_prediction_fraction["SitePredictionCount"] / top_site_prediction_fraction["TotalPredictions"]
    ) * 100.0
    top_site_prediction_fraction = top_site_prediction_fraction.rename(
        columns={
            "PEPTIDE": "TopPredictionSitePeptide",
            "PTM.Protein.Position": "TopPredictionSitePosition",
        }
    )

    dominant_ptm = (
        frame.groupby(["Protein.ID", "PTM.NAME"], dropna=False)
        .size()
        .rename("PTMCount")
        .reset_index()
        .sort_values(["Protein.ID", "PTMCount", "PTM.NAME"], ascending=[True, False, True])
        .groupby("Protein.ID", as_index=False)
        .first()
        .rename(columns={"PTM.NAME": "DominantPTMName", "PTMCount": "DominantPTMCount"})
    )
    dominant_ptm = dominant_ptm.merge(protein_prediction_counts.reset_index(), on="Protein.ID", how="left")
    dominant_ptm["DominantPTMSharePercent"] = (
        dominant_ptm["DominantPTMCount"] / dominant_ptm["TotalPredictions"]
    ) * 100.0

    protein_rollup = (
        protein_prediction_counts.reset_index()
        .merge(protein_unique_sites.reset_index(), on="Protein.ID", how="left")
        .merge(protein_unique_ptms.reset_index(), on="Protein.ID", how="left")
        .merge(top_site_by_protein, on="Protein.ID", how="left")
        .merge(
            top_site_prediction_fraction[
                [
                    "Protein.ID",
                    "TopPredictionSitePeptide",
                    "TopPredictionSitePosition",
                    "TopSitePredictionFraction",
                ]
            ],
            on="Protein.ID",
            how="left",
        )
        .merge(
            dominant_ptm[["Protein.ID", "DominantPTMName", "DominantPTMSharePercent"]],
            on="Protein.ID",
            how="left",
        )
        .sort_values(["TotalPredictions", "Protein.ID"], ascending=[False, True])
    )

    dominant_site_ptm_counts = (
        site_ptm_composition.sort_values(
            ["Protein.ID", "PEPTIDE", "PTM.Protein.Position", "RunCount", "PTM.NAME"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["Protein.ID", "PEPTIDE", "PTM.Protein.Position"], as_index=False)
        .first()[["Protein.ID", "PEPTIDE", "PTM.Protein.Position", "PTM.NAME", "RunCount"]]
        .rename(columns={"PTM.NAME": "DominantPTMName", "RunCount": "DominantPTMRunCount"})
    )
    site_disagreement = (
        site_ptm_composition.groupby(["Protein.ID", "PEPTIDE", "PTM.Protein.Position"], dropna=False)
        .agg(
            SiteRunCount=("RunCount", "sum"),
            SitePercentRuns=("PercentRuns", "max"),
            PTMNameCount=("PTM.NAME", "nunique"),
            PTMNames=("PTM.NAME", lambda values: " | ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    run_disagreement_summary = site_disagreement.merge(
        dominant_site_ptm_counts,
        on=["Protein.ID", "PEPTIDE", "PTM.Protein.Position"],
        how="left",
    )
    run_disagreement_summary["DominantPTMRunSharePercent"] = (
        run_disagreement_summary["DominantPTMRunCount"] / run_disagreement_summary["SiteRunCount"]
    ) * 100.0
    run_disagreement_summary["HasCompetingPTMs"] = run_disagreement_summary["PTMNameCount"] > 1
    run_disagreement_summary = run_disagreement_summary.sort_values(
        ["HasCompetingPTMs", "PTMNameCount", "SitePercentRuns", "Protein.ID", "PEPTIDE"],
        ascending=[False, False, False, True, True],
    )

    if "PTM.DeltaMass" in frame.columns and frame["PTM.DeltaMass"].notna().any():
        delta_mass_frame = frame[frame["PTM.DeltaMass"].notna()].copy()
        delta_mass_frame["RoundedDeltaMass"] = delta_mass_frame["PTM.DeltaMass"].round(4)
        dominant_mass_ptm = (
            delta_mass_frame.groupby(["RoundedDeltaMass", "PTM.NAME"], dropna=False)
            .size()
            .rename("PTMCount")
            .reset_index()
            .sort_values(["RoundedDeltaMass", "PTMCount", "PTM.NAME"], ascending=[True, False, True])
            .groupby("RoundedDeltaMass", as_index=False)
            .first()
            .rename(columns={"PTM.NAME": "DominantPTMName", "PTMCount": "DominantPTMCount"})
        )
        delta_mass_summary = (
            delta_mass_frame.groupby("RoundedDeltaMass", dropna=False)
            .agg(
                ObservationCount=("RoundedDeltaMass", "size"),
                UniqueProteins=("Protein.ID", "nunique"),
                UniqueSites=("PEPTIDE", "nunique"),
                PTMNameCount=("PTM.NAME", "nunique"),
            )
            .reset_index()
            .merge(dominant_mass_ptm, on="RoundedDeltaMass", how="left")
            .sort_values(["ObservationCount", "RoundedDeltaMass"], ascending=[False, True])
        )
        delta_mass_summary["DominantPTMSharePercent"] = (
            delta_mass_summary["DominantPTMCount"] / delta_mass_summary["ObservationCount"]
        ) * 100.0
    else:
        delta_mass_summary = _empty_table(
            [
                "RoundedDeltaMass",
                "ObservationCount",
                "UniqueProteins",
                "UniqueSites",
                "PTMNameCount",
                "DominantPTMName",
                "DominantPTMCount",
                "DominantPTMSharePercent",
            ]
        )

    return {
        "ptm_frequency_by_site": ptm_frequency_by_site,
        "top_ptms_per_protein": top_ptms_per_protein,
        "peptide_position_frequency": peptide_position_frequency,
        "site_ptm_composition": site_ptm_composition,
        "protein_rollup": protein_rollup,
        "run_disagreement_summary": run_disagreement_summary,
        "delta_mass_summary": delta_mass_summary,
    }


def _write_tables(tables: dict[str, pd.DataFrame], tables_dir: Path, timestamp: str) -> list[str]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, frame in tables.items():
        path = tables_dir / f"{name}_{timestamp}.csv"
        frame.to_csv(path, index=False)
        written.append(str(path))
    return written


def render_analysis_plots(
    tables: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    timestamp: str,
    top_k: int,
) -> list[str]:
    """Render deterministic PNG plots for the analysis report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save_current(path: Path) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        written.append(str(path))

    freq = tables["ptm_frequency_by_site"].head(top_k)
    if not freq.empty:
        plt.figure(figsize=(14, 6))
        freq_labels = [
            f"{protein} | {peptide} | {ptm} @{position}"
            for protein, peptide, ptm, position in zip(
                freq["Protein.ID"],
                freq["PEPTIDE"],
                freq["PTM.NAME"],
                freq["PTM.Protein.Position"],
                strict=True,
            )
        ]
        plt.barh(freq_labels, freq["PercentRuns"], color="#33658A")
        plt.xlabel("% of Runs")
        plt.ylabel("Protein | Peptide | PTM @ Position")
        plt.title(f"Top {len(freq)} PTM calls by recurrence")
        save_current(plots_dir / f"ptm_frequency_by_site_{timestamp}.png")

    top_ptms = tables["top_ptms_per_protein"].head(top_k)
    if not top_ptms.empty:
        plt.figure(figsize=(14, 6))
        top_labels = [
            f"{protein} | {ptm} @{position}"
            for protein, ptm, position in zip(
                top_ptms["Protein.ID"],
                top_ptms["PTM.NAME"],
                top_ptms["PTM.Protein.Position"],
                strict=True,
            )
        ]
        plt.barh(top_labels, top_ptms["PercentRuns"], color="#2F855A")
        plt.xlabel("% of Runs")
        plt.ylabel("Protein | PTM @ Position")
        plt.title(f"Top {len(top_ptms)} PTM calls across proteins")
        save_current(plots_dir / f"top_ptms_per_protein_{timestamp}.png")

    pp_freq = tables["peptide_position_frequency"].head(top_k)
    if not pp_freq.empty:
        plt.figure(figsize=(14, 6))
        pp_labels = [
            f"{protein} | {peptide} @{position}"
            for protein, peptide, position in zip(
                pp_freq["Protein.ID"],
                pp_freq["PEPTIDE"],
                pp_freq["PTM.Protein.Position"],
                strict=True,
            )
        ]
        plt.barh(pp_labels, pp_freq["PercentRuns"], color="#805AD5")
        plt.xlabel("% of Runs")
        plt.ylabel("Protein | Peptide @ Position")
        plt.title(f"Top {len(pp_freq)} peptide-position sites")
        save_current(plots_dir / f"peptide_position_frequency_{timestamp}.png")

    site_comp = tables["site_ptm_composition"]
    top_sites = tables["peptide_position_frequency"].head(min(top_k, 10))
    if not top_sites.empty and not site_comp.empty:
        selected = set(
            zip(top_sites["Protein.ID"], top_sites["PEPTIDE"], top_sites["PTM.Protein.Position"], strict=True)
        )
        filtered = site_comp[
            site_comp.apply(
                lambda row: (row["Protein.ID"], row["PEPTIDE"], row["PTM.Protein.Position"]) in selected,
                axis=1,
            )
        ].copy()
        filtered["SiteLabel"] = filtered.apply(
            lambda row: f"{row['Protein.ID']} | {row['PEPTIDE']} @{row['PTM.Protein.Position']}",
            axis=1,
        )
        pivot = filtered.pivot_table(
            index="SiteLabel",
            columns="PTM.NAME",
            values="PercentRuns",
            aggfunc="sum",
            fill_value=0,
        )
        plt.figure(figsize=(14, 6))
        bottom = None
        for column in pivot.columns:
            values = pivot[column]
            if bottom is None:
                plt.bar(pivot.index, values, label=column)
                bottom = values.copy()
            else:
                plt.bar(pivot.index, values, bottom=bottom, label=column)
                bottom = bottom + values
        plt.ylabel("% of Runs")
        plt.xlabel("Protein | Peptide @ Position")
        plt.xticks(rotation=60, ha="right")
        plt.title("PTM composition across top sites")
        plt.legend(title="PTM.NAME", fontsize=8)
        save_current(plots_dir / f"site_composition_{timestamp}.png")

    heatmap_frame = tables["peptide_position_frequency"].head(top_k).copy()
    if not heatmap_frame.empty:
        heatmap_frame["SiteLabel"] = heatmap_frame.apply(
            lambda row: f"{row['PEPTIDE']} @{row['PTM.Protein.Position']}",
            axis=1,
        )
        heatmap_data = heatmap_frame.pivot_table(
            index="Protein.ID",
            columns="SiteLabel",
            values="PercentRuns",
            aggfunc="max",
            fill_value=0,
        )
        if not heatmap_data.empty:
            plt.figure(figsize=(max(8, len(heatmap_data.columns) * 1.2), max(4, len(heatmap_data.index) * 1.2)))
            plt.imshow(heatmap_data.values, aspect="auto", cmap="YlGnBu")
            plt.colorbar(label="% of Runs")
            plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=60, ha="right")
            plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
            plt.xlabel("Peptide @ Position")
            plt.ylabel("Protein.ID")
            plt.title("Top recurrent sites heatmap")
            save_current(plots_dir / f"site_heatmap_{timestamp}.png")

    delta_mass = tables["delta_mass_summary"].head(top_k)
    if not delta_mass.empty:
        plt.figure(figsize=(14, 6))
        labels = [f"{mass:.4f}" for mass in delta_mass["RoundedDeltaMass"]]
        plt.barh(labels, delta_mass["ObservationCount"], color="#C05621")
        plt.xlabel("Observation Count")
        plt.ylabel("Rounded Delta Mass")
        plt.title(f"Top {len(delta_mass)} recurring delta masses")
        save_current(plots_dir / f"delta_mass_summary_{timestamp}.png")

    protein_ptm = (
        tables["ptm_frequency_by_site"]
        .groupby(["Protein.ID", "PTM.NAME"], dropna=False)["RunCount"]
        .sum()
        .reset_index(name="RunCountSum")
    )
    if not protein_ptm.empty:
        protein_pivot = protein_ptm.pivot_table(
            index="Protein.ID",
            columns="PTM.NAME",
            values="RunCountSum",
            aggfunc="sum",
            fill_value=0,
        )
        plt.figure(figsize=(14, 6))
        bottom = None
        for column in protein_pivot.columns:
            values = protein_pivot[column]
            if bottom is None:
                plt.bar(protein_pivot.index, values, label=column)
                bottom = values.copy()
            else:
                plt.bar(protein_pivot.index, values, bottom=bottom, label=column)
                bottom = bottom + values
        plt.ylabel("Summed Run Recurrence")
        plt.xlabel("Protein.ID")
        plt.xticks(rotation=45, ha="right")
        plt.title("PTM composition by protein")
        plt.legend(title="PTM.NAME", fontsize=8)
        save_current(plots_dir / f"protein_ptm_composition_{timestamp}.png")

    return written


def _metrics_from_frame(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "total_predictions": int(len(frame)),
        "unique_proteins": int(frame["Protein.ID"].nunique()),
        "unique_peptides": int(frame["PEPTIDE"].nunique()),
        "unique_ptm_names": int(frame["PTM.NAME"].nunique()),
        "unique_sites": int(frame[["Protein.ID", "PEPTIDE", "PTM.Protein.Position"]].drop_duplicates().shape[0]),
        "unique_site_ptm_combinations": int(
            frame[["Protein.ID", "PEPTIDE", "PTM.Protein.Position", "PTM.NAME"]].drop_duplicates().shape[0]
        ),
    }


def _heuristics(table: pd.DataFrame, *, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    high = table[table["PercentRuns"] >= HIGH_CONSENSUS_THRESHOLD].head(top_k)
    low = table[table["PercentRuns"] < LOW_CONSENSUS_THRESHOLD].sort_values("PercentRuns", ascending=True).head(top_k)
    return cast(list[dict[str, Any]], high.to_dict(orient="records")), cast(
        list[dict[str, Any]], low.to_dict(orient="records")
    )


def _protein_focus_summary(table: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    focused = table[table["TopSitePredictionFraction"] >= PROTEIN_FOCUS_THRESHOLD].head(top_k)
    return cast(list[dict[str, Any]], focused.to_dict(orient="records"))


def _disagreement_sites(table: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    disagreements = table[table["HasCompetingPTMs"]].head(top_k)
    return cast(list[dict[str, Any]], disagreements.to_dict(orient="records"))


def _delta_mass_hotspots(table: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], table.head(top_k).to_dict(orient="records")) if not table.empty else []


def write_analysis_report(
    *,
    report_dir: Path,
    timestamp: str,
    analysis_mode: str,
    source: dict[str, str | None],
    metrics: dict[str, int],
    num_runs: int,
    tables: dict[str, pd.DataFrame],
    table_files: list[str],
    plot_files: list[str],
) -> dict[str, Any]:
    """Write the timestamped JSON and Markdown report files."""
    summary_json_path = report_dir / f"summary_{timestamp}.json"
    summary_md_path = report_dir / f"summary_{timestamp}.md"
    high_consensus_sites, low_consensus_sites = _heuristics(tables["ptm_frequency_by_site"], top_k=10)
    disagreement_sites = _disagreement_sites(tables["run_disagreement_summary"], top_k=10)
    protein_focus_summary = _protein_focus_summary(tables["protein_rollup"], top_k=10)
    delta_mass_hotspots = _delta_mass_hotspots(tables["delta_mass_summary"], top_k=10)

    payload = {
        "analysis_timestamp": timestamp,
        "analysis_mode": analysis_mode,
        "source": source,
        "output": {
            "report_dir": str(report_dir),
            "summary_markdown": str(summary_md_path),
            "summary_json": str(summary_json_path),
            "plots": plot_files,
            "tables": table_files,
        },
        "metrics": metrics,
        "num_runs": num_runs,
        "high_consensus_sites": high_consensus_sites,
        "low_consensus_sites": low_consensus_sites,
        "disagreement_sites": disagreement_sites,
        "protein_focus_summary": protein_focus_summary,
        "delta_mass_hotspots": delta_mass_hotspots,
    }

    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)

    lines = [
        f"# PTM Run Analysis - {timestamp}",
        "",
        f"- Analysis mode: `{analysis_mode}`",
        f"- Number of runs represented: `{num_runs}`",
        f"- Source run dir: `{source.get('run_dir')}`",
        f"- Source aggregated CSV: `{source.get('aggregated_csv')}`",
        "",
        "## Summary Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")

    lines.extend(
        [
            "",
            "## Deterministic Highlights",
            "",
            f"- High-consensus sites: `{len(high_consensus_sites)}`",
            f"- Disagreement sites: `{len(disagreement_sites)}`",
            f"- Focused proteins: `{len(protein_focus_summary)}`",
            f"- Delta-mass hotspots: `{len(delta_mass_hotspots)}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
        ]
    )
    lines.extend(f"- `{Path(path).name}`" for path in table_files)
    lines.extend(
        [
            "",
            "## Generated Plots",
            "",
        ]
    )
    if plot_files:
        lines.extend(f"- `{Path(path).name}`" for path in plot_files)
    else:
        lines.append("- None (plot generation disabled)")

    with open(summary_md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return payload


def analyze_run_results(
    *,
    run_dir: str | None = None,
    aggregated_csv: str | None = None,
    top_k: int = 25,
    output_dir: str | None = None,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Analyze a completed PTM run or aggregated CSV and write a timestamped report."""
    resolved = resolve_analysis_input(run_dir=run_dir, aggregated_csv=aggregated_csv)
    timestamp = format_timestamp()

    if resolved["analysis_mode"] == "run_dir":
        raw_frame = load_run_predictions(resolved["run_dir"])
    else:
        raw_frame = load_aggregated_predictions(resolved["aggregated_csv"])

    frame = _normalize_predictions(raw_frame)
    tables = build_summary_tables(frame, top_k=top_k)

    parent = Path(output_dir) if output_dir is not None else resolved["output_parent"]
    report_dir = parent / f"analysis_{timestamp}"
    tables_dir = report_dir / "tables"
    plots_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)

    table_files = _write_tables(tables, tables_dir, timestamp)
    plot_files = render_analysis_plots(tables, plots_dir, timestamp=timestamp, top_k=top_k) if create_plots else []

    return write_analysis_report(
        report_dir=report_dir,
        timestamp=timestamp,
        analysis_mode=resolved["analysis_mode"],
        source={
            "run_dir": str(resolved["run_dir"]) if resolved["run_dir"] is not None else None,
            "aggregated_csv": str(resolved["aggregated_csv"]) if resolved["aggregated_csv"] is not None else None,
            "input_files": resolved["input_files"],
        },
        metrics=_metrics_from_frame(frame),
        num_runs=max(int(frame["Run.Index"].nunique()), 1),
        tables=tables,
        table_files=table_files,
        plot_files=plot_files,
    )
