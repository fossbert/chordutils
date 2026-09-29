"""Reconstruction of regimen episodes from the MSK-CHORD treatment timeline.

MSK-CHORD reports systemic therapy as one row per agent (``data_timeline_treatment.txt``).
``pivot_therapies`` turns these rows into non-overlapping regimen episodes per patient; the
remaining objects in this module are its building blocks.
"""

import numpy as np
import pandas as pd
from itertools import groupby

# Separator between an agent name and the internal bookkeeping suffix used by pivot_therapies
# (row index and split counters, e.g. 'GEMCITABINE#3-1'). It must not occur in agent names;
# MSK-CHORD agent names contain letters, digits, spaces, '-', '(', ')', ':' and "'" but no '#'.
# Digits cannot be used as delimiter because some agent names end in digits (e.g. 'SODIUM IODIDE I-131').
ID_SEP = "#"

def pivot_therapies(d: pd.DataFrame,
                    min_size:int=7,
                    verbose: bool=False,
                    agent:str="AGENT",
                    start_date:str="START_DATE",
                    stop_date:str="STOP_DATE",
                    investigative:str="RX_INVESTIGATIVE",
                    subtype:str="SUBTYPE"):

    """Convert the treatment timeline of ONE patient from one row per agent into regimen episodes.

    MSK-CHORD reports systemic therapy as one row per agent with a start and stop day. Regimens
    (e.g. FOLFIRINOX) are therefore not given explicitly and have to be reconstructed from the
    overlap of the individual agents. This function cuts the patient's timeline into
    non-overlapping episodes, each labelled with the set of agents given during that episode.

    Algorithm
    ---------
    Rows are processed in chronological order (stable sort by ``start_date``). Every day in
    ``[start, stop]`` counts as a treatment day, i.e. both boundaries are included. For each new
    agent row and every episode already collected:

    1. **Overlap** -> the overlapping days become a combination episode ``<existing>_<new>``.
    2. **Remainder of the existing episode** (days not covered by the new agent):

       - no days left -> the existing episode is removed (fully absorbed into the combination),
       - one consecutive block left -> the existing episode is shortened to that block,
       - several blocks left (new agent lies *within* the existing episode) -> the existing
         episode is split; blocks shorter than ``min_size`` days are DISCARDED.
    3. **Remainder of the new agent** (days not covered by any existing episode) becomes an
       episode of the new agent alone. If these days fall apart into several blocks (only
       possible if rows are not sorted by start), each block is kept separately; blocks shorter
       than ``min_size`` days are discarded as in 2.

    Finally, agent names are cleaned from bookkeeping suffixes and sorted alphabetically, so a
    regimen is always written the same way (e.g. 'FLUOROURACIL_IRINOTECAN_LEUCOVORIN_OXALIPLATIN').

    Example: gemcitabine days 0-100 and nab-paclitaxel days 0-80 give two episodes,
    'GEMCITABINE_PACLITAXEL PROTEIN-BOUND' (days 0-80) and 'GEMCITABINE' (days 81-100).

    Parameters
    ----------
    d : pd.DataFrame
        Treatment timeline rows of a single patient (use with ``groupby('PATIENT_ID').apply``).
    min_size : int, default 7
        Minimum length in days of a block that is kept when an episode has to be split into
        several non-consecutive blocks (see step 2). Short fragments of this kind are usually
        artefacts of slightly misaligned start/stop dates rather than real treatment periods.
        Note that a single consecutive remainder is always kept, irrespective of its length.
    verbose : bool, default False
        Print every bookkeeping step (useful to follow the algorithm for one patient).
    agent, start_date, stop_date, investigative, subtype : str
        Column names of agent name, start day, stop day, investigational flag ('Y'/'N') and
        treatment subtype (e.g. 'Chemo', 'Immuno').

    Returns
    -------
    pd.DataFrame
        One row per episode with columns ``agent`` (sorted agent names joined by '_'),
        ``start``, ``stop`` (both included) and ``days`` (= stop - start + 1), sorted by start.

    Notes
    -----
    - Investigational agents are anonymised in MSK-CHORD (AGENT = 'INVESTIGATIVE'). They are
      labelled 'INVESTIGATIVE:<SUBTYPE>' (e.g. 'INVESTIGATIVE:Chemo'). Two overlapping
      investigational rows therefore appear twice in the name, which is intended: they
      are (most likely) two different study drugs.
    - Overlapping rows of the *same* named agent are likewise treated as two agents and appear
      twice in the regimen name (e.g. 'CISPLATIN_CISPLATIN'); they are not merged.
    - Output episodes do not overlap in time. Days on which no agent was given (treatment
      breaks) are not part of any episode.
    """

    # Chronological order is required for a well-defined result (see step 3 in the docstring).
    # A stable sort keeps the original order of rows with equal start days.
    d = d.sort_values(start_date, kind="stable")

    treats = {} #This is our book keeping dictionary which is modifed throughout the function

    for i, row in enumerate(d.itertuples()):

        ag_str = getattr(row, agent)

        if getattr(row, investigative) == 'Y':

            ag_str = ag_str + ':' + getattr(row, subtype)

        # The row index makes the key unique while agents are being combined and split;
        # it is removed again by Therapy.clean_agent at the end.
        ther = Therapy(f"{ag_str}{ID_SEP}{i}", getattr(row, start_date), getattr(row, stop_date))

        if verbose: 
            print(f"Testing treatment: {ther.agent}")

        if i==0: # first line will always just be added.

            if verbose:
                print(f"Adding treatment: {ther.agent}")

            treats[ther.agent] = ther

        else:

            if verbose:
                print(f"Getting copy #{i} of treatments with {len(treats)} items")

            new_range = ther.time_range #this is needed for removing any treatments overlapping later on

            treats_current = treats.copy() # here the current version our treatments is copied

            for k, v in treats_current.items():        

                # This part is for cleaning the existing values of the results

                a = np.setdiff1d(v.time_range, ther.time_range)

                if len(a)>0:

                    if arr_is_consecutive(a): # the above setdiff may split treatments in non-consecutive time ranges

                        if verbose:
                            print(f"Pruning existing treatment: {v.agent}")

                        new_start = np.min(a)

                        new_stop = np.max(a)

                        treats[v.agent] = Therapy(v.agent, new_start, new_stop)                    

                    else:

                        if verbose:
                            print(f"treatment: {v.agent} in two time ranges, adjusting ...")

                        sub_ranges = arr_split_consecutive(a, min_size=min_size)

                        for n, sr in enumerate(sub_ranges, 1):

                            new_start = np.min(sr)

                            new_stop = np.max(sr)

                            new_agent = f"{v.agent}-{n}"

                            if verbose:
                                print(f"Adding {new_agent} as treatment")

                            treats[new_agent] = Therapy(new_agent, new_start, new_stop)

                        if verbose:
                            print(f"Removing parent treatment: {v.agent}") 

                        del treats[v.agent]      

                else:
                    
                    if verbose:
                        print(f"Deleting treatment: {v.agent}")

                    del treats[v.agent]

                #### Overlapping part ####

                ab = np.intersect1d(v.time_range, ther.time_range)

                if len(ab)>0:

                    combo = v.agent + '_' + ther.agent

                    new_start = np.min(ab)

                    new_stop = np.max(ab)

                    combo_ther = Therapy(combo, new_start, new_stop)

                    if verbose:
                        print(f"Adding combination: {combo}")

                    treats[combo_ther.agent] = combo_ther
            
                new_range =  np.setdiff1d(new_range, v.time_range)

            if len(new_range)>0: #Important for this to be outside the for loop above so all current agents are tested against the new agent

                if arr_is_consecutive(new_range):

                    if verbose:
                        print(f"Adding treatment: {ther.agent}")

                    treats[ther.agent] = Therapy(ther.agent, np.min(new_range), np.max(new_range))

                else:
                    # The uncovered days of the new agent fall apart into several blocks. Using
                    # min/max here would create an episode that overlaps existing ones.
                    for n, sr in enumerate(arr_split_consecutive(new_range, min_size=min_size), 1):

                        new_agent = f"{ther.agent}-{n}"

                        if verbose:
                            print(f"Adding {new_agent} as treatment")

                        treats[new_agent] = Therapy(new_agent, np.min(sr), np.max(sr))

    if not treats:
        return pd.DataFrame(columns=["agent", "start", "stop", "days"])

    for v in treats.values():

        v.clean_agent() # removes bookkeeping suffixes and sorts agent names

    treats_df = treat_dict_to_frame(treats)

    return treats_df.sort_values("start", ignore_index=True)


def treat_dict_to_frame(treat_dict: dict, sort_starts:bool=True):

    """Helper function to process lines of therapy"""

    dout = pd.concat([v.to_frame() for v in treat_dict.values()], axis=0, ignore_index=True)

    if sort_starts:

        dout = dout.sort_values('start', ignore_index=True)

    return dout


class Therapy:

    """This class facilitates collection and processing of important information on lines of therapy."""

    def __init__(self, agent:str, start:int, stop:int):

        self.agent = str(agent)
        self.start = int(start)
        self.stop = int(stop)
        self.time_range = np.arange(self.start, self.stop + 1) # This means that stop dates ARE included
        self.days = len(self.time_range)

    def __repr__(self):

        return f"Therapy({self.agent}, {self.start},  {self.stop}, {self.days})"

    def to_frame(self):

        dout = {k:v for k, v in vars(self).items() if k != "time_range"}

        return pd.DataFrame(data=dout, index=[0])

    def clean_agent(self):

        """Remove bookkeeping suffixes ('#<row>' and split counters '-1', '-2', ...) and sort agents.

        E.g. 'OXALIPLATIN#12-1_FLUOROURACIL#3' -> 'FLUOROURACIL_OXALIPLATIN'. Only the part from
        ``ID_SEP`` onwards is removed, so agent names ending in digits ('SODIUM IODIDE I-131')
        stay intact.
        """

        parts = self.agent.split("_")

        self.agent = '_'.join(sorted(p.split(ID_SEP, 1)[0] for p in parts))



def arr_is_consecutive(arr: np.ndarray):

    """Tests time ranges on whether they contain consecutive integers. 

    Returns:
        True/False
    """

    return (np.diff(arr)==1).all()


def arr_split_consecutive(arr: np.ndarray, min_size:int):

    """Splits a non-consecutive integer array (time range) on each break.

    Returns:
        lout: list containing all consecutive subarrays with a size of at least min_size consecutive intergers.
    """

    out = []
    
    for _, g in groupby(enumerate(arr), lambda x: x[0] - x[1]):
        
        out.append([v for _, v in g])

    lout = [o for o in out if len(o)>=min_size]

    return lout
