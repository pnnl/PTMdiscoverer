"""Utility helpers for PTMdiscoverer.

Delta-mass extraction, prompt building, JSON parsing, protein metadata
loading, position augmentation, and PTM record validation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence  # noqa: I001
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def extract_delta_masses(
    protein_id: str,
    msrun_table: pd.DataFrame,
    base_path: str = "",
    *,
    min_probability: float = 0.9,
    min_delta_mass: float = 10.0,
    select_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return filtered / de-duplicated delta-mass rows for one protein.

    Parameters:
    - protein_id: Identifier expected in `Protein.ID` column.
    - msrun_table: DataFrame with column `MSRUNID` listing run identifiers.
    - base_path: Root directory containing `Results_MSFragger/<run>/psm.tsv` files.
    - min_probability: Minimum probability threshold (strict > comparison).
    - min_delta_mass: Minimum delta mass (strict > comparison) to treat as modified.
    - select_columns: Optional explicit column list for the returned frame.

    Processing Summary:
        1. Iterate MS run IDs, attempt to read each `psm.tsv` (tab separated).
        2. Normalize columns: spaces replaced with dots for consistency.
        3. Filter: (a) zero missed cleavages, (b) Probability > min_probability,
           (c) Protein.ID matches target.
        4. Keep rows with Delta.Mass > min_delta_mass.
        5. Localization confidence filters: Number.Best.Positions == 1 AND
           Ions.Best.Position >= Ions.Second.Best.Position AND
           Score.Best.Position > Score.Second.Best.Position.
        6. Round Delta.Mass to 3 decimals, sort for determinism, drop duplicates,
           compute retention time minutes (RT).
        7. Derive absolute protein position if Protein.Start is present.

    Returns:
        DataFrame (possibly empty) with curated modification events.

    Limitations / Gotchas:
        * Only the FIRST numeric token in Best.Positions is used (multi-site strings like
          "3,7 (S,T)" are simplified to 3).
        * Does not disambiguate peptides occurring multiple times in a protein; relies on
          MSFragger's Protein.Start to reflect the chosen mapping.
        * Silent (now warned) skips occur for missing psm.tsv files per run.
        * If localization columns are absent or renamed, function will raise KeyError.
        * Delta mass rounding (3 decimals) may merge near-isobaric events.
        * Probability and scoring thresholds are strict (>), excluding exact boundary values.
        * Returns empty DataFrame instead of raising when no qualifying data found.
    """
    psm_frames: list[pd.DataFrame] = []

    # Gather candidate rows across runs
    if "MSRUNID" not in msrun_table.columns:
        logger.warning("extract_delta_masses: msrun_table missing 'MSRUNID' column; returning empty DataFrame.")
        return pd.DataFrame()

    for msrunid in msrun_table["MSRUNID"]:
        psm_path = os.path.join(base_path, f"Results_MSFragger/{msrunid}/psm.tsv")
        try:
            tb = pd.read_csv(psm_path, sep="\t")
        except FileNotFoundError:
            logger.warning("extract_delta_masses: psm.tsv not found for run %s at %s", msrunid, psm_path)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_delta_masses: failed reading %s (%s)", psm_path, exc)
            continue

        tb.columns = tb.columns.str.replace(" ", ".", regex=False)

        tb = tb[
            (tb["Number.of.Missed.Cleavages"] == 0)
            & (tb["Probability"] > min_probability)
            & (tb["Protein.ID"] == protein_id)
        ]
        if tb.empty:
            logger.debug(
                "extract_delta_masses: no qualifying rows after filters for run %s (protein %s)",
                msrunid,
                protein_id,
            )
            continue

        tb["MSRUNID"] = msrunid
        psm_frames.append(tb)

    psms = pd.concat(psm_frames, ignore_index=True) if psm_frames else pd.DataFrame()

    if psms.empty:
        logger.info(
            "extract_delta_masses: no PSMs gathered for protein %s across %d runs", protein_id, len(msrun_table)
        )
        return pd.DataFrame()

    modified = psms[psms["Delta.Mass"] > min_delta_mass]
    # Track localization pruning for transparency
    before_loc = len(modified)
    loc_mask = (
        (modified["Number.Best.Positions"] == 1)
        & (modified["Ions.Best.Position"] >= modified["Ions.Second.Best.Position"])
        & (modified["Score.Best.Position"] > modified["Score.Second.Best.Position"])
    )
    pruned = before_loc - int(loc_mask.sum())
    modified = modified[loc_mask]
    if pruned > 0:
        logger.warning(
            "extract_delta_masses: pruned %d row(s) due to multi-site or weaker localization (protein %s)",
            pruned,
            protein_id,
        )
    if modified.empty:
        logger.info(
            "extract_delta_masses: all candidate rows filtered out by delta mass or localization criteria (protein %s)",
            protein_id,
        )
        return pd.DataFrame()

    modified["Delta.Mass"] = modified["Delta.Mass"].round(3)

    modified = modified.sort_values(
        by=[
            "Peptide",
            "Charge",
            "Delta.Mass",
            "Best.Positions",
            "MSRUNID",
            "Score.Best.Position",
        ],
        ascending=[True, True, True, True, True, False],
    ).drop_duplicates(subset=["Peptide", "Charge", "Delta.Mass", "Best.Positions", "MSRUNID"])

    modified["RT"] = modified["Retention"] / 60.0

    # Derive modification position within full protein sequence when Protein.Start present.
    # Best.Positions examples: "15 M", "7 S", "3,7 (S,T)" -> extract first integer as representative site.
    if "Protein.Start" in modified.columns:
        # Protein.Start is 1-based, Best.Positions positional integers are 1-based within peptide.
        with_start = modified["Protein.Start"].astype(int)
        best_pos_index = modified["Best.Positions"].astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(float)
        modified["Best.Positions.Index"] = best_pos_index
        # Absolute coordinate = peptide start + (peptide-relative position - 1)
        modified["Positions.In.Protein"] = (with_start + best_pos_index - 1).astype(float)
    else:
        logger.debug(
            "extract_delta_masses: 'Protein.Start' absent; cannot derive Positions.In.Protein for protein %s",
            protein_id,
        )
        modified["Best.Positions.Index"] = float("nan")
        modified["Positions.In.Protein"] = float("nan")

    if select_columns is None:
        select_columns = (
            "Spectrum",
            "MSRUNID",
            "RT",
            "Peptide",
            "Is.Unique",
            "Charge",
            "Calculated.Peptide.Mass",
            "Calculated.M/Z",
            "Delta.Mass",
            "Positions.In.Protein",
            "Best.Positions",
            "Probability",
            "Score.Best.Position",
        )
    return modified[list(select_columns)]


# ---------------------------------------------------------------------------
# Inference helper utilities
# ---------------------------------------------------------------------------


def load_protein_metadata(path: str) -> list[dict[str, Any]]:
    """Return list of protein metadata dicts loaded from a JSON file.

    Entries that are not dicts or lack a string ``uniprot_id`` are
    dropped with a warning log.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    proteins = data.get("proteins", [])
    if not isinstance(proteins, list):
        raise ValueError("'proteins' key must map to a list in metadata JSON")
    valid: list[dict[str, Any]] = []
    for i, entry in enumerate(proteins):
        if not isinstance(entry, dict):
            logger.warning("load_protein_metadata: entry %d is not a dict, skipping", i)
            continue
        uid = entry.get("uniprot_id")
        if not isinstance(uid, str) or not uid:
            logger.warning("load_protein_metadata: entry %d missing or invalid 'uniprot_id', skipping", i)
            continue
        valid.append(entry)
    if len(valid) < len(proteins):
        logger.warning("load_protein_metadata: kept %d of %d entries after validation", len(valid), len(proteins))
    return valid


def build_prompt(base_prompt: str, protein_name: str, delta_mass_df: pd.DataFrame) -> str:
    """Build LLM prompt with peptide delta-mass table (drops 'Spectrum' column)."""
    df_for_prompt = delta_mass_df.drop(columns=[c for c in ("Spectrum",) if c in delta_mass_df.columns])
    header = ",".join(df_for_prompt.columns)
    rows = "\n".join(df_for_prompt.astype(str).agg("\t".join, axis=1))
    return (
        f"{base_prompt}\nTable of peptides and delta masses detected in samples for {protein_name}:\n{header}\n{rows}"
    )


def parse_first_json_block(answer: str) -> list[dict[str, Any]]:
    """Parse first fenced JSON block from LLM answer into list of dict records."""
    match = re.search(r"```json\s*(.*?)\s*```", answer, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    try:
        data = json.loads(match.group(1).strip())
    except Exception:  # noqa: BLE001
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def augment_protein_positions(records: list[dict[str, Any]], delta_mass_df: pd.DataFrame) -> None:
    """Augment or correct PTM.Protein.Position using extracted delta-mass DataFrame.

    Mapping logic:
        Build (Peptide uppercase, peptide-local site index) -> set(abs positions) based on
        columns 'Peptide', 'Best.Positions', 'Positions.In.Protein'. Only apply when the set
        resolves to exactly one integer and the record either lacks the field or has a non-positive value.
        Simple fallback: if peptide maps uniquely to one absolute coordinate across table, use it.
    """
    if delta_mass_df.empty:
        return
    expected = {"Peptide", "Best.Positions", "Positions.In.Protein"}
    if not expected.issubset(delta_mass_df.columns):
        return

    mapping: dict[tuple[str, int], set[int]] = {}
    peptide_only: dict[str, set[int]] = {}
    for _, row in delta_mass_df.iterrows():
        pep = str(row["Peptide"]).upper()
        best = str(row["Best.Positions"])  # e.g. 'W11'
        m = re.search(r"(\d+)", best)
        if not m:
            continue
        local_idx = int(m.group(1))
        try:
            abs_pos_f = float(row["Positions.In.Protein"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("augment: non-numeric Positions.In.Protein for peptide %s: %s", pep, exc)
            continue
        if abs_pos_f <= 0 or abs_pos_f != int(abs_pos_f):
            continue
        abs_pos = int(abs_pos_f)
        mapping.setdefault((pep, local_idx), set()).add(abs_pos)
        peptide_only.setdefault(pep, set()).add(abs_pos)

    def needs_update(val: Any) -> bool:
        try:
            return int(val) <= 0
        except (ValueError, TypeError):  # int() can raise either for bad input
            return True

    unresolved = 0
    for rec in records:
        pep_key = str(rec.get("PEPTIDE", "")).upper()
        if not pep_key:
            continue
        current = rec.get("PTM.Protein.Position")
        if current is not None and not needs_update(current):
            continue
        sites = rec.get("PTM.Best.Positions")
        if not sites:
            continue
        site_raw = sites[0] if isinstance(sites, list) and sites else sites
        m = re.search(r"(\d+)", str(site_raw))
        if not m:
            continue
        local_idx = int(m.group(1))
        candidates = mapping.get((pep_key, local_idx))
        if candidates and len(candidates) == 1:
            rec["PTM.Protein.Position"] = next(iter(candidates))
            continue
        pep_abs = peptide_only.get(pep_key)
        if pep_abs and len(pep_abs) == 1:
            rec["PTM.Protein.Position"] = next(iter(pep_abs))
            continue
        # Ambiguous or unresolved mapping – leave untouched; will be dropped later if invalid
        unresolved += 1
    if unresolved:
        logger.warning(
            "augment_protein_positions: %d record(s) remained ambiguous/unresolved (no unique absolute position)",
            unresolved,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("This module provides library functions only.")


# ---------------------------------------------------------------------------
# LLM output validation
# ---------------------------------------------------------------------------

#: Required fields and their allowed types for a valid PTM prediction record.
_REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "PEPTIDE": (str,),
    "PTM.DeltaMass": (int, float),
    "PTM.Best.Positions": (list,),
    "PTM.NAME": (str,),
    "PTM.Impact.In.Enzyme.Function": (str,),
    "PTM.Protein.Position": (int, float),
}


def validate_ptm_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only records that have all required fields with correct types.

    Records missing any key from ``_REQUIRED_FIELDS`` or having a value that
    does not match the expected type(s) are dropped with a warning log.
    """
    valid: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        ok = True
        for field, types in _REQUIRED_FIELDS.items():
            if field not in rec:
                logger.warning("validate_ptm_records: record %d missing required field '%s'", i, field)
                ok = False
                break
            if not isinstance(rec[field], types):
                logger.warning(
                    "validate_ptm_records: record %d field '%s' has type %s, expected %s",
                    i,
                    field,
                    type(rec[field]).__name__,
                    "/".join(t.__name__ for t in types),
                )
                ok = False
                break
        if ok:
            valid.append(rec)
    if len(valid) < len(records):
        logger.warning(
            "validate_ptm_records: dropped %d of %d records due to schema violations",
            len(records) - len(valid),
            len(records),
        )
    return valid
