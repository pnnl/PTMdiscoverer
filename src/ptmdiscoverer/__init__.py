"""Top-level PTMdiscoverer package exports."""

from .inference import (  # noqa: F401
    InferenceContext,
    assess_study_folder,
    format_timestamp,
    run_and_aggregate_multiple_inferences,
    run_inference,
    validate_study_folder,
)
from .reporting import analyze_run_results  # noqa: F401
from .utils import (  # noqa: F401
    augment_protein_positions,
    build_prompt,
    extract_delta_masses,
    load_protein_metadata,
    parse_first_json_block,
)

__all__ = [
    # Extraction
    "extract_delta_masses",
    # Inference entry points
    "assess_study_folder",
    "run_inference",
    "run_and_aggregate_multiple_inferences",
    "validate_study_folder",
    "InferenceContext",
    "analyze_run_results",
    # Utilities
    "build_prompt",
    "parse_first_json_block",
    "augment_protein_positions",
    "load_protein_metadata",
    "format_timestamp",
]

# Tell pdoc not to duplicate documentation for re-exported symbols at the package root.
# This keeps the authoritative docs under their defining modules (utils / inference).
__pdoc__: dict[str, bool] = {
    name: False
    for name in __all__  # hide the re-export listings while keeping import convenience
}
