"""Reading the MSK-CHORD study files (cBioPortal format).

The MSK-CHORD download (``msk_chord_2024``) is a cBioPortal study folder:

- ``data_clinical_sample.txt`` / ``data_clinical_patient.txt``: one row per sample / patient.
  The first four lines start with '#' and contain, per column, the display name, a description
  (for NLP-derived variables incl. source and handling of missing data), the data type and a
  display priority. The fifth line holds the column names.
- ``data_timeline_<name>.txt``: event tables without header lines. All days (START_DATE,
  STOP_DATE) are relative to the date of the patient's FIRST sequenced sample (day 0).
- Genomic files (``data_mutations.txt``, ``data_cna.txt``, ``data_sv.txt``, ...).

``ChordStudy`` gives cached access to these files so that every analysis step reads the raw
data the same way.
"""

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]

# Line prefixes of the cBioPortal clinical header block, in file order.
_CLINICAL_HEADER_FIELDS = ["display_name", "description", "datatype", "priority"]


def _count_comment_lines(path: PathLike, prefix: str = "#") -> int:
    """Number of leading lines starting with ``prefix`` (cBioPortal meta-header)."""

    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith(prefix):
                break
            n += 1
    return n


def read_cbio_table(path: PathLike, **kwargs) -> pd.DataFrame:
    """Read a cBioPortal data file, skipping its '#' meta-header lines if present.

    The number of header lines is detected from the file rather than hard-coded, and
    ``comment='#'`` is deliberately NOT used because '#' may occur inside values.

    Parameters
    ----------
    path : str or Path
        File to read (tab-separated).
    **kwargs
        Passed on to ``pd.read_table`` (e.g. ``usecols``, ``low_memory``).
    """

    return pd.read_table(path, skiprows=_count_comment_lines(path), **kwargs)


def read_clinical_dictionary(path: PathLike) -> pd.DataFrame:
    """Data dictionary of a clinical file, built from its four '#' header lines.

    Returns
    -------
    pd.DataFrame
        Index = column name; columns ``display_name``, ``description``, ``datatype``,
        ``priority``. The descriptions document how NLP-derived variables were generated and
        what a missing value means, which is worth citing in a methods section.
    """

    n = _count_comment_lines(path)
    with open(path, encoding="utf-8") as fh:
        lines = [next(fh).rstrip("\n") for _ in range(n + 1)]

    columns = lines[-1].split("\t")
    meta = {field: line.lstrip("#").split("\t")
            for field, line in zip(_CLINICAL_HEADER_FIELDS, lines[:-1])}
    return pd.DataFrame(meta, index=pd.Index(columns, name="column"))


class ChordStudy:
    """Cached access to the files of one MSK-CHORD study folder.

    Every table is read once and then served from memory; callers always receive a copy, so
    modifying a returned DataFrame never changes the cache.

    Parameters
    ----------
    path : str or Path
        The study folder (containing ``meta_study.txt``).

    Examples
    --------
    >>> study = ChordStudy("MSK_CHORD_2024")
    >>> study.timelines                     # available timeline tables
    >>> tx = study.timeline("treatment")    # data_timeline_treatment.txt
    >>> study.dictionary("sample").loc["TUMOR_PURITY", "description"]
    """

    def __init__(self, path: PathLike):

        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"MSK-CHORD study folder not found: {self.path}")

    def __repr__(self):
        return f"ChordStudy({str(self.path)!r})"

    # -- file level ---------------------------------------------------------------------------

    def file(self, name: str) -> Path:
        """Full path of a file in the study folder; raises if it does not exist."""

        p = self.path / name
        if not p.exists():
            raise FileNotFoundError(f"{name} not found in {self.path}")
        return p

    @lru_cache(maxsize=None)
    def _read(self, name: str, usecols: Optional[tuple] = None) -> pd.DataFrame:
        return read_cbio_table(self.file(name), usecols=list(usecols) if usecols else None, low_memory=False)

    def read(self, name: str, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Read (and cache) any tab-separated file of the study folder by file name."""

        return self._read(name, tuple(usecols) if usecols else None).copy()

    # -- clinical tables ----------------------------------------------------------------------

    def sample(self) -> pd.DataFrame:
        """``data_clinical_sample.txt``: one row per sequenced sample."""

        return self.read("data_clinical_sample.txt")

    def patient(self) -> pd.DataFrame:
        """``data_clinical_patient.txt``: one row per patient (incl. OS_MONTHS / OS_STATUS)."""

        return self.read("data_clinical_patient.txt")

    def dictionary(self, table: str = "sample") -> pd.DataFrame:
        """Data dictionary of the 'sample' or 'patient' table (see ``read_clinical_dictionary``)."""

        if table not in ("sample", "patient"):
            raise ValueError(f"table must be 'sample' or 'patient', got {table!r}")
        return read_clinical_dictionary(self.file(f"data_clinical_{table}.txt"))

    # -- timelines ----------------------------------------------------------------------------

    @property
    def timelines(self) -> list:
        """Names of all available timeline tables (e.g. 'treatment', 'diagnosis', ...)."""

        return sorted(p.stem.removeprefix("data_timeline_") for p in self.path.glob("data_timeline_*.txt"))

    def timeline(self, name: str) -> pd.DataFrame:
        """Timeline table ``data_timeline_<name>.txt``; days are relative to the first sequencing."""

        if name not in self.timelines:
            raise ValueError(f"Unknown timeline {name!r}. Available: {', '.join(self.timelines)}")
        return self.read(f"data_timeline_{name}.txt")


def pickle_transfer(path:str, type:str="in", out_object=None, verbose: bool=False):

    """Load (``type='in'``) or save (``type='out'``) a pickled dictionary of DataFrames."""

    if type not in ["in", "out"]:
        raise ValueError(f"type can only be 'in' or 'out, got {type}")

    if type == "in":

        with open(path, "rb") as inpick:
            inpick = pickle.load(inpick)
            if verbose:
                print(f"Imported pickle file from {path}")
        return inpick

    else:
        if out_object:
            with open(path, "wb") as outpick:
                pickle.dump(out_object, outpick)
                if verbose:
                    print(f"File got pickled to {path}")
        else:
            print(f"out_object cannot be empty/None if type is {type}. Please reconsider.")
