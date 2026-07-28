"""Table structure recognition adapters and normalization."""

from .analyzer import (
    analyze_table_block,
    analyze_table_blocks,
    reconcile_reclassified_table_blocks,
)
from .description import generate_table_description

__all__ = [
    "analyze_table_blocks",
    "analyze_table_block",
    "reconcile_reclassified_table_blocks",
    "generate_table_description",
]
