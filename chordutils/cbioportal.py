"""Driver annotations from cBioPortal exports -- moved to the ``cbiokit`` package.

Kept as a thin alias so that ``chordutils.cbioportal.<name>`` keeps working; the
documentation, export format notes and tests live in ``cbiokit.alterations``.
"""

from cbiokit.alterations import (AlterationExport, driver_event_matrix, driver_gene_matrix,  # noqa: F401
                                 panel_genes_from_export, read_alteration_export)
from cbiokit.alterations import DRIVER_TAG, NO_ALTERATION, NOT_PROFILED, TYPES  # noqa: F401
