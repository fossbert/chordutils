"""Entity configurations (one ``EntityConfig`` per tumour entity).

To add an entity, create ``<entity>.py`` with an ``EntityConfig`` and import it here.
"""

from .pdac import PDAC

__all__ = ["PDAC"]
