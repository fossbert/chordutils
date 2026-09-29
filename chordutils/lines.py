"""Lines of therapy from regimen episodes.

``tables.treatment_episodes`` (via ``pivot_therapies``) cuts a patient's systemic therapy into
non-overlapping episodes, one per combination of agents. A clinical *line* of therapy usually
spans several episodes: agents are added in cycle 2, dropped for toxicity (FOLFIRINOX ->
FOLFIRI), or continued as maintenance (-> 5-FU). ``build_lines_of_therapy`` merges episodes
into lines with explicit, configurable rules (``config.LineRules``).

Adapted from an earlier PDAC analysis module (chordendpoints); protocol names are entity-specific and
therefore live in the entity config.
"""

from typing import Iterable, Union

import pandas as pd

from .config import LineRules

ID = "PATIENT_ID"


def split_regimen(regimen: str, sep: str = "_") -> frozenset:
    """Agents of a regimen string, e.g. 'FLUOROURACIL_LEUCOVORIN' -> {'FLUOROURACIL', 'LEUCOVORIN'}."""

    return frozenset(a for a in str(regimen).split(sep) if a)


def normalize_agents(agents: Iterable[str], rules: LineRules) -> frozenset:
    """Agents as used for line decisions: ignored agents removed, equivalent agents unified."""

    return frozenset(rules.equivalent_agents.get(a, a) for a in agents if a not in rules.ignored_agents)


def classify_regimen(agents: Union[str, Iterable[str]], rules: LineRules = LineRules()) -> str:
    """Protocol name of a set of agents.

    Parameters
    ----------
    agents : str or iterable of str
        Regimen string ('A_B_C') or agent names.
    rules : LineRules
        ``regimen_names`` (first exact match wins) and ``ignored_agents``.

    Returns
    -------
    str
        The protocol name, else the sorted agents joined by '+'. Investigational agents
        ('INVESTIGATIVE:<subtype>') are reported as 'TRIAL + <backbone>' (or 'TRIAL' alone).
        'UNKNOWN' if nothing but ignored agents remains.
    """

    if isinstance(agents, str):
        agents = split_regimen(agents)

    core = {a for a in agents if a not in rules.ignored_agents}
    trial = {a for a in core if a.startswith("INVESTIGATIVE")}
    backbone = core - trial

    name = next((n for n, s in rules.regimen_names if backbone == set(s)), None)
    if name is None:
        name = "+".join(sorted(backbone))

    if trial:
        return f"TRIAL + {name}" if name else "TRIAL"
    return name or "UNKNOWN"


LINE_COLUMNS = [ID, "LINE", "LINE_START", "LINE_END", "REGIMEN", "AGENTS_INITIAL", "AGENTS_ALL",
                "N_EPISODES", "DEESCALATED"]


def build_lines_of_therapy(episodes: pd.DataFrame, rules: LineRules = LineRules(),
                           regimen_col: str = "REGIMEN", start_col: str = "START_DATE",
                           stop_col: str = "STOP_DATE") -> pd.DataFrame:
    """Combine regimen episodes into lines of therapy.

    Per patient, episodes are processed in chronological order. An episode CONTINUES the
    current line if the gap since the end of the line is at most ``rules.gap_days`` AND

    - it starts within ``rules.grace_days`` of the line start (its agents are then added to
      the initial regimen), OR
    - its agents are a subset of the initial regimen (de-escalation, maintenance,
      re-escalation, switch between ``equivalent_agents``), compared after
      ``normalize_agents``.

    Otherwise it starts a new line: a new agent after the grace period, or a treatment-free
    interval longer than ``gap_days``.

    Parameters
    ----------
    episodes : pd.DataFrame
        Output of ``tables.treatment_episodes`` (PATIENT_ID, REGIMEN, START_DATE, STOP_DATE).
    rules : LineRules
        Usually ``config.lines``.

    Returns
    -------
    pd.DataFrame
        One row per line: PATIENT_ID, LINE (1, 2, ...), LINE_START, LINE_END, REGIMEN (protocol
        name of the initial regimen), AGENTS_INITIAL, AGENTS_ALL (all agents given in the line),
        N_EPISODES, DEESCALATED (the last episode had fewer agents than the initial regimen).

    Notes
    -----
    Limitations of rule-based lines: a planned switch within a protocol (e.g. sequential
    'AC -> T' in breast cancer, where paclitaxel follows doxorubicin/cyclophosphamide) is
    a new agent after the grace period and therefore becomes a new line. Such schedules need
    an additional, entity-specific rule.
    """

    d = episodes.sort_values([ID, start_col, stop_col], kind="stable")
    rows = []

    for pid, gb in d.groupby(ID, sort=False):

        current = None

        for ep in gb.itertuples(index=False):

            agents = split_regimen(getattr(ep, regimen_col))
            start, stop = int(getattr(ep, start_col)), int(getattr(ep, stop_col))

            if current is not None:
                in_grace = start <= current["LINE_START"] + rules.grace_days
                gap_ok = (start - current["LINE_END"]) <= rules.gap_days
                subset = normalize_agents(agents, rules) <= normalize_agents(current["AGENTS_INITIAL"], rules)

                if gap_ok and (in_grace or subset):
                    if in_grace:
                        current["AGENTS_INITIAL"] = current["AGENTS_INITIAL"] | agents
                    current["AGENTS_ALL"] = current["AGENTS_ALL"] | agents
                    current["LINE_END"] = max(current["LINE_END"], stop)
                    current["N_EPISODES"] += 1
                    current["_last"] = agents
                    continue

                rows.append(current)

            current = {ID: pid, "LINE_START": start, "LINE_END": stop, "AGENTS_INITIAL": agents,
                       "AGENTS_ALL": agents, "N_EPISODES": 1, "_last": agents}

        if current is not None:
            rows.append(current)

    if not rows:
        return pd.DataFrame(columns=LINE_COLUMNS)

    lines = pd.DataFrame(rows)
    lines["DEESCALATED"] = [normalize_agents(l, rules) < normalize_agents(i, rules)
                            for l, i in zip(lines["_last"], lines["AGENTS_INITIAL"])]
    lines["REGIMEN"] = [classify_regimen(a, rules) for a in lines["AGENTS_INITIAL"]]
    lines["AGENTS_INITIAL"] = ["_".join(sorted(a)) for a in lines["AGENTS_INITIAL"]]
    lines["AGENTS_ALL"] = ["_".join(sorted(a)) for a in lines["AGENTS_ALL"]]
    lines["LINE"] = lines.groupby(ID).cumcount() + 1

    return lines[LINE_COLUMNS].sort_values([ID, "LINE"], ignore_index=True)
